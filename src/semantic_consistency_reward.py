import os
import sys

import torch
import torchaudio
from torch import nn


class SemanticConsistencyReward:
    def __init__(self):
        musicfm_repo_path = "YOUR_LOCAL_MUSICFM_REPO_PATH"
        if not os.path.exists(musicfm_repo_path):
            raise FileNotFoundError(
                f"MusicFM repo path {musicfm_repo_path} does not exist. "
                "Please clone the source musicfmrepo from "
                "https://github.com/minzwon/musicfm and set the path in the code."
            )
        musicfm_sample_rate = 24000
        # add the repo path to the sys.path so that we can import the MusicFM25Hz module
        # one level up from the repo path
        sys.path.append(os.path.dirname(musicfm_repo_path))
        from musicfm.model.musicfm_25hz import MusicFM25Hz

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.musicfm = MusicFM25Hz(
            is_flash=False,
            stat_path=os.path.join(musicfm_repo_path, "data", "msd_stats.json"),
            model_path=os.path.join(musicfm_repo_path, "data", "pretrained_msd.pt"),
        )
        self.musicfm.eval()
        self.musicfm.to(self.device)
        self.model_sample_rate = musicfm_sample_rate

    def __call__(self, src_wavs, src_sr):
        with torch.no_grad():
            wavs = []
            for wav_i, src_sr_i in zip(src_wavs, src_sr, strict=False):
                if src_sr_i != self.model_sample_rate:
                    wav_i = torchaudio.transforms.Resample(
                        src_sr_i, self.model_sample_rate
                    )(wav_i)
                wavs.append(wav_i)
            wav = torch.stack(wavs)
            wav = wav.to(self.device)
            assert wav.ndim == 2 or wav.ndim == 3, (
                f"Waveform must be 2 or 3 dimensional, got {wav.ndim}"
            )
            if wav.ndim == 3:
                wav = wav.squeeze(1)
            unmasked_logits, unmasked_hidden_emb = self.musicfm.get_predictions(wav)
            quantizer_name = "melspec_2048"
            unmasked_semantic_consistency = (
                nn.functional.softmax(unmasked_logits[quantizer_name], dim=-1)
                .max(dim=-1)
                .values.mean(dim=-1)
                .cpu()
                .numpy()
            )

            logits, hidden_emb, losses, accuracies, probs_of_masked = self.musicfm(wav)
            # multiplying by masking rate to compensate for differences in
            # prediction hardness
            bs, seq_len, _ = logits[quantizer_name].shape
            masking_rate_per_sample = [len(x) / seq_len for x in probs_of_masked]
            masked_semantic_consistency_per_sample = [
                x.mean() * masking_rate_per_sample[i]
                for i, x in enumerate(probs_of_masked)
            ]
            masked_semantic_consistency = (
                torch.stack(masked_semantic_consistency_per_sample).cpu().numpy()
            )

            return [
                {
                    "unmasked_semantic_consistency": float(
                        unmasked_semantic_consistency[i]
                    ),
                    "masked_semantic_consistency": float(
                        masked_semantic_consistency[i]
                    ),
                }
                for i in range(bs)
            ]
