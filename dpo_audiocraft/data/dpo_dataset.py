import copy
import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from audiocraft.data.audio_dataset import (
    DEFAULT_EXTS,
    AudioDataset,
    BaseInfo,
    SegmentInfo,
    _resolve_audio_meta,
    audio_read,
    convert_audio,
    find_audio_files,
    logger,
)
from audiocraft.data.info_audio_dataset import AudioInfo, clusterify_all_meta
from audiocraft.data.music_dataset import MusicInfo, augment_music_info_description
from audiocraft.data.zip import PathInZip
from audiocraft.modules.conditioners import (
    JointEmbedCondition,
    SegmentWithAttributes,
    WavCondition,
)
from torch.nn import functional as F


@dataclass(order=True)
class AudioPairMeta(BaseInfo):
    path: str
    duration: float
    sample_rate: int

    # negative sample for DPO
    negative_path: str
    negative_duration: float
    negative_sample_rate: int

    amplitude: float | None = None
    weight: float | None = None
    # info_path is used to load additional information about the audio file that is stored in zip files.
    info_path: PathInZip | None = None

    @classmethod
    def from_dict(cls, dictionary: dict):
        base = cls._dict2fields(dictionary)
        if "info_path" in base and base["info_path"] is not None:
            base["info_path"] = PathInZip(base["info_path"])
        return cls(**base)

    def to_dict(self):
        d = super().to_dict()
        if d["info_path"] is not None:
            d["info_path"] = str(d["info_path"])
        return d


def load_audio_pair_meta(path: str | Path, resolve: bool = True, fast: bool = True) -> list[AudioPairMeta]:
    open_fn = gzip.open if str(path).lower().endswith(".gz") else open
    with open_fn(path, "rb") as fp:  # type: ignore
        lines = fp.readlines()
    meta = []
    for line in lines:
        d = json.loads(line)
        m = AudioPairMeta.from_dict(d)
        if resolve:
            m = _resolve_audio_meta(m, fast=fast)
        meta.append(m)
    return meta


class DPOAudioDataset(AudioDataset):
    def __init__(
        self,
        meta: list[AudioPairMeta],
        segment_duration: float | None = None,
        shuffle: bool = True,
        num_samples: int = 10_000,
        sample_rate: int = 48_000,
        channels: int = 2,
        pad: bool = True,
        sample_on_duration: bool = True,
        sample_on_weight: bool = True,
        min_segment_ratio: float = 0.5,
        max_read_retry: int = 10,
        return_info: bool = False,
        min_audio_duration: float | None = None,
        max_audio_duration: float | None = None,
        shuffle_seed: int = 0,
        load_wav: bool = True,
        permutation_on_files: bool = False,
    ):
        assert len(meta) > 0, "No audio meta provided to AudioDataset. Please check loading of audio meta."
        assert segment_duration is None or segment_duration > 0
        assert segment_duration is None or min_segment_ratio >= 0
        self.segment_duration = segment_duration
        self.min_segment_ratio = min_segment_ratio
        self.max_audio_duration = max_audio_duration
        self.min_audio_duration = min_audio_duration
        if self.min_audio_duration is not None and self.max_audio_duration is not None:
            assert self.min_audio_duration <= self.max_audio_duration
        self.meta: list[AudioPairMeta] = self._filter_duration(meta)
        assert len(self.meta)  # Fail fast if all data has been filtered.
        self.total_duration = sum(d.duration for d in self.meta)

        if segment_duration is None:
            num_samples = len(self.meta)
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.channels = channels
        self.pad = pad
        self.sample_on_weight = sample_on_weight
        self.sample_on_duration = sample_on_duration
        self.sampling_probabilities = self._get_sampling_probabilities()
        self.max_read_retry = max_read_retry
        self.return_info = return_info
        self.shuffle_seed = shuffle_seed
        self.current_epoch: int | None = None
        self.load_wav = load_wav
        if not load_wav:
            assert segment_duration is not None
        self.permutation_on_files = permutation_on_files
        if permutation_on_files:
            assert not self.sample_on_duration
            assert not self.sample_on_weight
            assert self.shuffle

    def read_audio_file(self, file_path: str, duration: float, rng: torch.Generator):
        max_seek = max(0, duration - self.segment_duration * self.min_segment_ratio)
        seek_time = torch.rand(1, generator=rng).item() * max_seek
        wav, sr = audio_read(file_path, seek_time, self.segment_duration, pad=False)
        wav = convert_audio(wav, sr, self.sample_rate, self.channels)
        n_frames = wav.shape[-1]
        target_frames = int(self.segment_duration * self.sample_rate)
        if self.pad:
            wav = F.pad(wav, (0, target_frames - n_frames))

        return wav, seek_time, n_frames

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, SegmentInfo]:
        assert self.segment_duration is not None
        rng = torch.Generator()
        if self.shuffle:
            # We use index, plus extra randomness, either totally random if we don't know the epoch.
            # otherwise we make use of the epoch number and optional shuffle_seed.
            if self.current_epoch is None:
                rng.manual_seed(index + self.num_samples * random.randint(0, 2**24))
            else:
                rng.manual_seed(index + self.num_samples * (self.current_epoch + self.shuffle_seed))
        else:
            # We only use index
            rng.manual_seed(index)

        for retry in range(self.max_read_retry):
            try:
                file_meta: AudioPairMeta = self.sample_file(index, rng)
                wav, seek_time, n_frames = self.read_audio_file(file_meta.path, file_meta.duration, rng)
                negative_wav, _, _ = self.read_audio_file(file_meta.negative_path, file_meta.negative_duration, rng)
                segment_info = SegmentInfo(
                    file_meta,
                    seek_time,
                    n_frames=n_frames,
                    total_frames=n_frames,
                    sample_rate=self.sample_rate,
                    channels=wav.shape[0],
                )
            except Exception as exc:
                logger.warning("Error opening file %s: %r", file_meta.path, exc)
                if retry == self.max_read_retry - 1:
                    raise
            else:
                break

        if self.return_info:
            # Returns the wav and additional information on the wave segment
            return wav, negative_wav, segment_info
        else:
            return wav, negative_wav

    def collater(self, samples):
        """The collater function has to be provided to the dataloader
        if AudioDataset has return_info=True in order to properly collate
        the samples of a batch.
        """
        if self.segment_duration is None and len(samples) > 1:
            assert self.pad, "Must allow padding when batching examples of different durations."

        # In this case the audio reaching the collater is of variable length as segment_duration=None.
        to_pad = self.segment_duration is None and self.pad
        if to_pad:
            max_len = max([wav.shape[-1] for wav, _ in samples])

            def _pad_wav(wav):
                return F.pad(wav, (0, max_len - wav.shape[-1]))

        if self.return_info:
            if len(samples) > 0:
                assert len(samples[0]) == 3
                assert isinstance(samples[0][0], torch.Tensor)
                assert isinstance(samples[0][1], torch.Tensor)
                assert isinstance(samples[0][2], SegmentInfo)

            wavs = []
            negative_wavs = []
            segment_infos = []
            for wav, negative_wav, info in samples:
                wavs.append(wav)
                negative_wavs.append(negative_wav)
                segment_infos.append(copy.deepcopy(info))

            if to_pad:
                # Each wav could be of a different duration as they are not segmented.
                for i in range(len(samples)):
                    # Determines the total length of the signal with padding, so we update here as we pad.
                    segment_infos[i].total_frames = max_len
                    wavs[i] = _pad_wav(wavs[i])
                    negative_wavs[i] = _pad_wav(negative_wavs[i])
            wav = torch.stack(wavs)
            negative_wav = torch.stack(negative_wavs)
            return wav, negative_wav, segment_infos
        else:
            assert isinstance(samples[0], torch.Tensor)
            assert isinstance(samples[1], torch.Tensor)
            if to_pad:
                samples[0] = _pad_wav(samples[0])
                samples[1] = _pad_wav(samples[1])
            return torch.stack(samples[0]), torch.stack(samples[1])

    @classmethod
    def from_meta(cls, root: str | Path, **kwargs):
        """Instantiate AudioDataset from a path to a directory containing a manifest as a jsonl file.

        Args:
            root (str or Path): Path to root folder containing audio files.
            kwargs: Additional keyword arguments for the AudioDataset.
        """
        root = Path(root)
        if root.is_dir():
            if (root / "data.jsonl").exists():
                root = root / "data.jsonl"
            elif (root / "data.jsonl.gz").exists():
                root = root / "data.jsonl.gz"
            else:
                raise ValueError(
                    "Don't know where to read metadata from in the dir. "
                    "Expecting either a data.jsonl or data.jsonl.gz file but none found."
                )
        meta = load_audio_pair_meta(root)
        return cls(meta, **kwargs)

    @classmethod
    def from_path(
        cls,
        root: str | Path,
        minimal_meta: bool = True,
        exts: list[str] = DEFAULT_EXTS,
        **kwargs,
    ):
        """Instantiate AudioDataset from a path containing (possibly nested) audio files.

        Args:
            root (str or Path): Path to root folder containing audio files.
            minimal_meta (bool): Whether to only load minimal metadata or not.
            exts (list of str): Extensions for audio files.
            kwargs: Additional keyword arguments for the AudioDataset.
        """
        root = Path(root)
        if root.is_file():
            meta = load_audio_pair_meta(root, resolve=True)
        else:
            meta = find_audio_files(root, exts, minimal=minimal_meta, resolve=True)
        return cls(meta, **kwargs)


class DPOInfoAudioDataset(DPOAudioDataset):
    def __init__(self, meta: list[AudioPairMeta], **kwargs):
        super().__init__(clusterify_all_meta(meta), **kwargs)

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, SegmentWithAttributes]:
        if not self.return_info:
            wav = super().__getitem__(index)
            assert isinstance(wav, torch.Tensor)
            return wav
        wav, negative_wav, meta = super().__getitem__(index)
        return wav, negative_wav, AudioInfo(**meta.to_dict())


class DPOMusicDataset(DPOInfoAudioDataset):
    def __init__(
        self,
        *args,
        info_fields_required: bool = False,
        merge_text_p: float = 0.0,
        drop_desc_p: float = 0.0,
        drop_other_p: float = 0.0,
        joint_embed_attributes: list[str] = [],
        paraphrase_source: str | None = None,
        paraphrase_p: float = 0,
        **kwargs,
    ):
        kwargs["return_info"] = True  # We require the info for each song of the dataset.
        super().__init__(*args, **kwargs)
        self.info_fields_required = info_fields_required
        self.merge_text_p = merge_text_p
        self.drop_desc_p = drop_desc_p
        self.drop_other_p = drop_other_p
        self.joint_embed_attributes = joint_embed_attributes
        self.paraphraser = None

    def __getitem__(self, index):
        wav, negative_wav, info = super().__getitem__(index)
        info_data = info.to_dict()
        music_info_path = Path(info.meta.path).with_suffix(".json")

        if Path(music_info_path).exists():
            with open(music_info_path) as json_file:
                music_data = json.load(json_file)
                music_data.update(info_data)
                music_info = MusicInfo.from_dict(music_data, fields_required=self.info_fields_required)
            if self.paraphraser is not None:
                music_info.description = self.paraphraser.sample(music_info.meta.path, music_info.description)
            if self.merge_text_p:
                music_info = augment_music_info_description(
                    music_info, self.merge_text_p, self.drop_desc_p, self.drop_other_p
                )
        else:
            music_info = MusicInfo.from_dict(info_data, fields_required=False)

        music_info.self_wav = WavCondition(
            wav=wav[None],
            length=torch.tensor([info.n_frames]),
            sample_rate=[info.sample_rate],
            path=[info.meta.path],
            seek_time=[info.seek_time],
        )

        for att in self.joint_embed_attributes:
            att_value = getattr(music_info, att)
            joint_embed_cond = JointEmbedCondition(
                wav[None],
                [att_value],
                torch.tensor([info.n_frames]),
                sample_rate=[info.sample_rate],
                path=[info.meta.path],
                seek_time=[info.seek_time],
            )
            music_info.joint_embed[att] = joint_embed_cond

        return wav, negative_wav, music_info
