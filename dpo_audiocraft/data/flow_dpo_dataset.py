# This file is based on implementations from audiocraft's datasets:
# https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/data/*.py
# In order to use it,
# this file should be copied into your local audiocraft repo to ../audiocraft/data/flow_dpo_dataset.py

import bisect
import gzip
import json
import math
import os
import pickle
import typing as tp
from pathlib import Path

import torch
from audiocraft.data.dpo_dataset import DPOMusicDataset
from audiocraft.data.jasco_dataset import JascoInfo

from ..modules.conditioners import SymbolicCondition
from ..utils.utils import construct_frame_chords
from .dpo_dataset import AudioPairMeta
from .music_dataset import MusicInfo

try:
    import dora
except ImportError:
    dora = None  # type: ignore


def _resolve_audio_pair_meta(m: AudioPairMeta, fast: bool = True) -> AudioPairMeta:
    def is_abs(m):
        if fast:
            return str(m)[0] == "/"
        else:
            os.path.isabs(str(m))

    if not dora:
        return m

    if not is_abs(m.path):
        m.path = dora.git_save.to_absolute_path(m.path)
    if m.info_path is not None and not is_abs(m.info_path.zip_path):
        m.info_path.zip_path = dora.git_save.to_absolute_path(m.path)
    return m


def load_audio_pair_meta(path: str | Path, resolve: bool = True, fast: bool = True) -> list[AudioPairMeta]:
    open_fn = gzip.open if str(path).lower().endswith(".gz") else open
    with open_fn(path, "rb") as fp:  # type: ignore
        lines = fp.readlines()
    meta = []
    for line in lines:
        d = json.loads(line)
        m = AudioPairMeta.from_dict(d)
        if resolve:
            m = _resolve_audio_pair_meta(m, fast=fast)
        meta.append(m)
    return meta


class FlowDPODataset(DPOMusicDataset):
    @classmethod
    def from_meta(cls, root: str | Path, **kwargs):
        """Instantiate AudioDataset from a path to a directory containing a manifest as a jsonl file.

        Args:
            root (str or Path): Path to root folder containing audio files.
            kwargs: Additional keyword arguments for the AudioDataset.
        """
        root = Path(root)
        # a directory is given
        if root.is_dir():
            if (root / "data.jsonl").exists():
                meta_json = root / "data.jsonl"
            elif (root / "data.jsonl.gz").exists():
                meta_json = root / "data.jsonl.gz"
            else:
                raise ValueError(
                    "Don't know where to read metadata from in the dir. "
                    "Expecting either a data.jsonl or data.jsonl.gz file but none found."
                )
        # jsonl file was specified
        else:
            assert root.exists() and root.suffix == ".jsonl", (
                "Either specified path not exist or it is not a jsonl format"
            )
            meta_json = root
            root = root.parent
        meta = load_audio_pair_meta(meta_json)
        kwargs["root"] = root
        return cls(meta, **kwargs)

    def __init__(
        self,
        *args,
        chords_card: int = 194,
        compression_model_framerate: float = 50.0,
        melody_kwargs: dict[str, tp.Any] | None = {},
        **kwargs,
    ):
        root = kwargs.pop("root")
        super().__init__(*args, **kwargs)

        chords_mapping_path = root / "chord_to_index_mapping.pkl"
        chords_path = root / "chords_per_track.pkl"
        self.mapping_dict = (
            pickle.load(open(chords_mapping_path, "rb")) if os.path.exists(chords_mapping_path) else None
        )

        self.chords_per_track = pickle.load(open(chords_path, "rb")) if os.path.exists(chords_path) else None

        self.compression_model_framerate = compression_model_framerate
        self.null_chord_idx = chords_card

        # self.melody_module = MelodyData(**melody_kwargs)  # type: ignore

    def _get_relevant_sublist(self, chords, timestamp):
        """
        Returns the sublist of chords within the specified timestamp and segment length.

        Args:
            chords (list): A sorted list of tuples containing (time changed, chord).
            timestamp (float): The timestamp at which to start the sublist.

        Returns:
            list: A list of chords within the specified timestamp and segment length.
        """
        end_time = timestamp + self.segment_duration

        # Use binary search to find the starting index of the relevant sublist
        start_index = bisect.bisect_left(chords, (timestamp,))

        if start_index != 0:
            prev_chord = chords[start_index - 1]
        else:
            prev_chord = (0.0, "N")

        relevant_chords = []

        for time_changed, chord in chords[start_index:]:
            if time_changed >= end_time:
                break
            relevant_chords.append((time_changed, chord))

        return relevant_chords, prev_chord

    def _get_chords(self, music_info: MusicInfo, effective_segment_dur: float) -> torch.Tensor:
        if self.chords_per_track is None:
            # use null chord when there's no chords in dataset
            seq_len = math.ceil(self.compression_model_framerate * effective_segment_dur)
            return torch.ones(seq_len, dtype=int) * self.null_chord_idx  # type: ignore

        fr = self.compression_model_framerate

        idx = music_info.meta.path.split("/")[-1].split(".")[0]
        chords = self.chords_per_track[idx]

        min_timestamp = music_info.seek_time

        chords = [(item[1], item[0]) for item in chords]
        chords, prev_chord = self._get_relevant_sublist(chords, min_timestamp)

        iter_min_timestamp = int(min_timestamp * fr) + 1

        frame_chords = construct_frame_chords(
            iter_min_timestamp,
            chords,
            self.mapping_dict,
            prev_chord[1],  # type: ignore
            fr,
            self.segment_duration,  # type: ignore
        )

        return torch.tensor(frame_chords)

    def __getitem__(self, index):
        wav, negative_wav, music_info = super().__getitem__(index)
        wav = wav.float()
        negative_wav = negative_wav.float()
        # downcast music info to jasco info
        jasco_info = JascoInfo(**{k: v for k, v in music_info.__dict__.items()})

        # get chords
        effective_segment_dur = (
            (wav.shape[-1] / self.sample_rate) if self.segment_duration is None else self.segment_duration
        )
        frame_chords = self._get_chords(music_info, effective_segment_dur)
        jasco_info.chords = SymbolicCondition(frame_chords=frame_chords)

        return wav, negative_wav, jasco_info
