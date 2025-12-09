# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import flashy
import torch
from omegaconf import DictConfig

from .. import models
from ..modules.conditioners import JascoCondConst, SegmentWithAttributes
from . import builders, jasco


class FlowDPOSolver(jasco.JascoSolver):
    """Solver for MR-FlowDPO"""

    DATASET_TYPE: builders.DatasetType = builders.DatasetType.FLOW_DPO

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # initialize generation parameters by config
        self.generation_params = {
            "cfg_coef_all": self.cfg.generate.lm.cfg_coef_all,
            "cfg_coef_txt": self.cfg.generate.lm.cfg_coef_txt,
        }

        self.latent_mean = cfg.compression_model_latent_mean
        self.latent_std = cfg.compression_model_latent_std
        self.mse = torch.nn.MSELoss(reduction="none")
        self._best_metric_name = "loss"
        self.beta = cfg.dpo.beta

    def build_model(self) -> None:
        super().build_model()

        # Create a frozen copy of the model as the reference model for DPO
        # Build a fresh model instance directly to get independent parameters
        # (shallow copy shares tensors, deepcopy fails with weight_norm)
        self.reference_model = models.builders.get_jasco_model(self.cfg, self.compression_model).to(self.device)
        self.reference_model.load_state_dict(self.model.state_dict())
        self.reference_model.eval()
        self.reference_model.requires_grad_(False)

    def _prepare_latents_and_attributes(
        self,
        batch: tuple[torch.Tensor, list[SegmentWithAttributes]],
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        audio, negative_audio, infos = batch
        audio = audio.to(self.device)
        negative_audio = negative_audio.to(self.device)
        assert audio.size(0) == len(infos), (
            f"Mismatch between number of items in audio batch ({audio.size(0)})",
            f" and in metadata ({len(infos)})",
        )

        latents = self._get_latents(audio)
        negative_latents = self._get_latents(negative_audio)

        # prepare attributes
        if JascoCondConst.CRD.value in self.cfg.conditioners:
            null_chord_idx = self.cfg.conditioners.chords.chords_emb.card
        else:
            null_chord_idx = -1
        attributes = [info.to_condition_attributes() for info in infos]
        if self.model.cfg_dropout is not None:
            attributes = self.model.cfg_dropout(
                samples=attributes,
                cond_types=["wav", "text", "symbolic"],
                null_chord_idx=null_chord_idx,
            )
        attributes = self.model.att_dropout(attributes)
        tokenized = self.model.condition_provider.tokenize(attributes)

        with self.autocast:
            condition_tensors = self.model.condition_provider(tokenized)

            # duplicate conditions for positive and negative samples
            condition_tensors_duplicate = self.duplicate_conditions(condition_tensors)

        return condition_tensors_duplicate, latents, negative_latents

    def _fm_loss_diff(self, v_theta: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        fm_loss = self.mse(v_theta, v)
        fm_loss = fm_loss.mean(dim=list(range(1, len(fm_loss.shape))))
        fm_loss_pos, fm_loss_neg = fm_loss.chunk(2)
        fm_loss_diffs = fm_loss_pos - fm_loss_neg
        return fm_loss_diffs

    def _compute_loss(self, v_theta: torch.Tensor, v_theta_ref: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        loss_diffs = self._fm_loss_diff(v_theta, v)
        loss_diffs_ref = self._fm_loss_diff(v_theta_ref, v)
        dpo_loss_per_pair = -torch.nn.functional.logsigmoid(-self.beta * (loss_diffs - loss_diffs_ref))
        return dpo_loss_per_pair.mean()

    def duplicate_conditions(self, condition_tensors: dict) -> dict:
        condition_tensors_duplicate = {}
        for cond, (cond_tensors, padding_mask) in condition_tensors.items():
            condition_tensors_duplicate[cond] = (
                cond_tensors.repeat(2, 1, 1),
                padding_mask.repeat(2, 1),
            )
        return condition_tensors_duplicate

    def run_step(
        self,
        idx: int,
        batch: tuple[torch.Tensor, list[SegmentWithAttributes]],
        metrics: dict,
    ) -> dict:
        """Perform one training or valid step on a given batch."""

        condition_tensors, latents, negative_latents = self._prepare_latents_and_attributes(batch)

        self.deadlock_detect.update("tokens_and_conditions")

        B, T, D = latents.shape
        assert negative_latents.shape == latents.shape
        device = self.device

        # normalize latents
        z_1 = self._normalized_latents(latents)
        z_1_negative = self._normalized_latents(negative_latents)

        # sample the N(0,1) prior
        z_0 = torch.randn(B, T, D, device=device)

        # random time parameter, between 0 to 1
        t = torch.rand((B, 1, 1), device=device)

        # interpolate data and prior
        # use the same time parameter and noise for both positive and negative samples
        z = self._z(z_0, z_1, t)
        z_negative = self._z(z_0, z_1_negative, t)
        z = torch.cat([z, z_negative], dim=0)

        # compute the GT vector field
        v = self._vector_field(z_0, z_1)
        v_negative = self._vector_field(z_0, z_1_negative)
        v = torch.cat([v, v_negative], dim=0)

        # duplicate time parameter for positive and negative samples
        t = t.repeat(2, 1, 1)
        with self.autocast:
            v_theta = self.model(latents=z, t=t, conditions=[], condition_tensors=condition_tensors)
            with torch.no_grad():
                v_theta_ref = self.reference_model(latents=z, t=t, conditions=[], condition_tensors=condition_tensors)
            loss = self._compute_loss(v_theta, v_theta_ref, v)
            unscaled_loss = loss.clone()

        self.deadlock_detect.update("loss")

        if self.is_training:
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            if self.scaler is not None:
                loss = self.scaler.scale(loss)
            self.deadlock_detect.update("scale")
            if self.cfg.fsdp.use:
                loss.backward()
                flashy.distrib.average_tensors(self.model.buffers())
            elif self.cfg.optim.eager_sync:
                with flashy.distrib.eager_sync_model(self.model):
                    loss.backward()
            else:
                # this should always be slower but can be useful
                # for weird use cases like multiple backwards.
                loss.backward()
                flashy.distrib.sync_model(self.model)
            self.deadlock_detect.update("backward")

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.cfg.optim.max_norm:
                if self.cfg.fsdp.use:
                    metrics["grad_norm"] = self.model.clip_grad_norm_(self.cfg.optim.max_norm)  # type: ignore
                else:
                    metrics["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optim.max_norm
                    )
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.deadlock_detect.update("optim")
            if self.scaler is not None:
                scale = self.scaler.get_scale()
                metrics["grad_scale"] = scale
            if not loss.isfinite().all():
                raise RuntimeError("Model probably diverged.")

        metrics["loss"] = unscaled_loss

        return metrics
