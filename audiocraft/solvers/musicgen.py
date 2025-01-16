# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
import logging
from pathlib import Path
import time
import typing as tp
import warnings

import flashy
import math
import omegaconf
import torch
from torch.nn import functional as F

from audiocraft.models.loaders import load_lm_model

from . import base, builders
from .compression import CompressionSolver
from .. import metrics as eval_metrics
from .. import models
from ..data.audio_dataset import AudioDataset
from ..data.music_dataset import MusicDataset, MusicInfo, AudioInfo
from ..data.sound_dataset import SoundDataset, SoundInfo
from ..data.audio_utils import normalize_audio
from ..modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition
from ..utils.cache import CachedBatchWriter, CachedBatchLoader
from ..utils.samples.manager import SampleManager
from ..utils.utils import get_dataset_from_loader, is_jsonable, warn_once, model_hash

logger = logging.getLogger(__name__)

class MusicGenSolver(base.StandardSolver):
    """Solver for MusicGen training task.

    Used in: https://arxiv.org/abs/2306.05284
    """
    DATASET_TYPE: builders.DatasetType = builders.DatasetType.MUSIC

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        # easier access to sampling parameters
        self.generation_params = {
            'use_sampling': self.cfg.generate.lm.use_sampling,
            'temp': self.cfg.generate.lm.temp,
            'top_k': self.cfg.generate.lm.top_k,
            'top_p': self.cfg.generate.lm.top_p,
        }
        self._best_metric_name: tp.Optional[str] = 'ce'

        self._cached_batch_writer = None
        self._cached_batch_loader = None
        if cfg.cache.path:
            if cfg.cache.write:
                self._cached_batch_writer = CachedBatchWriter(Path(cfg.cache.path))
                if self.cfg.cache.write_num_shards:
                    self.logger.warning("Multiple shard cache, best_metric_name will be set to None.")
                    self._best_metric_name = None
            else:
                self._cached_batch_loader = CachedBatchLoader(
                    Path(cfg.cache.path), cfg.dataset.batch_size, cfg.dataset.num_workers,
                    min_length=self.cfg.optim.updates_per_epoch or 1)
                self.dataloaders['original_train'] = self.dataloaders['train']
                self.dataloaders['train'] = self._cached_batch_loader  # type: ignore

    @staticmethod
    def get_eval_solver_from_sig(sig: str, dtype: tp.Optional[str] = None,
                                 device: tp.Optional[str] = None, autocast: bool = True,
                                 batch_size: tp.Optional[int] = None,
                                 override_cfg: tp.Optional[tp.Union[dict, omegaconf.DictConfig]] = None,
                                 **kwargs):
        """Mostly a convenience function around magma.train.get_solver_from_sig,
        populating all the proper param, deactivating EMA, FSDP, loading the best state,
        basically all you need to get a solver ready to "play" with in single GPU mode
        and with minimal memory overhead.

        Args:
            sig (str): signature to load.
            dtype (str or None): potential dtype, as a string, i.e. 'float16'.
            device (str or None): potential device, as a string, i.e. 'cuda'.
            override_cfg (dict or omegaconf.DictConfig or None): potential device, as a string, i.e. 'cuda'.
        """
        from audiocraft import train
        our_override_cfg: tp.Dict[str, tp.Any] = {'optim': {'ema': {'use': False}}}
        our_override_cfg['autocast'] = autocast
        if dtype is not None:
            our_override_cfg['dtype'] = dtype
        if device is not None:
            our_override_cfg['device'] = device
        if batch_size is not None:
            our_override_cfg['dataset'] = {'batch_size': batch_size}
        if override_cfg is None:
            override_cfg = {}
        override_cfg = omegaconf.OmegaConf.merge(
            omegaconf.DictConfig(override_cfg), omegaconf.DictConfig(our_override_cfg))  # type: ignore
        solver = train.get_solver_from_sig(
            sig, override_cfg=override_cfg,
            load_best=True, disable_fsdp=True,
            ignore_state_keys=['optimizer', 'ema'], **kwargs)
        solver.model.eval()
        return solver

    def get_formatter(self, stage_name: str) -> flashy.Formatter: # type: ignore
        return flashy.Formatter({ # type: ignore
            'lr': '.2E',
            'ce': '.3f',
            'ppl': '.3f',
            'grad_norm': '.3E',
        }, exclude_keys=['ce_q*', 'ppl_q*'])

    @property
    def best_metric_name(self) -> tp.Optional[str]:
        return self._best_metric_name

    def build_model(self) -> None:
        """Instantiate models and optimizer."""
        # we can potentially not use all quantizers with which the EnCodec model was trained
        # (e.g. we trained the model with quantizers dropout)
        self.compression_model = CompressionSolver.wrapped_model_from_checkpoint(
            self.cfg, self.cfg.compression_model_checkpoint, device=self.device)
        assert self.compression_model.sample_rate == self.cfg.sample_rate, (
            f"Compression model sample rate is {self.compression_model.sample_rate} but "
            f"Solver sample rate is {self.cfg.sample_rate}."
            )
        # ensure we have matching configuration between LM and compression model
        assert self.cfg.transformer_lm.card == self.compression_model.cardinality, (
            "Cardinalities of the LM and compression model don't match: ",
            f"LM cardinality is {self.cfg.transformer_lm.card} vs ",
            f"compression model cardinality is {self.compression_model.cardinality}"
        )
        assert self.cfg.transformer_lm.n_q == self.compression_model.num_codebooks, (
            "Numbers of codebooks of the LM and compression models don't match: ",
            f"LM number of codebooks is {self.cfg.transformer_lm.n_q} vs ",
            f"compression model numer of codebooks is {self.compression_model.num_codebooks}"
        )
        self.logger.info("Compression model has %d codebooks with %d cardinality, and a framerate of %d",
                         self.compression_model.num_codebooks, self.compression_model.cardinality,
                         self.compression_model.frame_rate)
        # instantiate LM model
        self.model: models.LMModel = models.builders.get_lm_model(self.cfg).to(self.device)

        if 'reference_model_statedict' in self.cfg and self.cfg.reference_model_statedict is not None:
            self.reference_model: tp.Optional[models.LMModel] = load_lm_model(self.cfg.reference_model_statedict, device=self.device)
            self.reference_model.eval() # should already be set anyway
            for param in self.reference_model.parameters():
                param.requires_grad = False # idk if i need this but whatever
                param.data = param.data.half()
        else:
            self.reference_model = None

        if self.cfg.fsdp.use:
            assert not self.cfg.autocast, "Cannot use autocast with fsdp"
            self.model = self.wrap_with_fsdp(self.model) # type: ignore
        self.register_ema('model')
        # initialize optimization
        self.optimizer = builders.get_optimizer(builders.get_optim_parameter_groups(self.model), self.cfg.optim)
        self.lr_scheduler = builders.get_lr_scheduler(self.optimizer, self.cfg.schedule, self.total_updates)
        self.register_stateful('model', 'optimizer', 'lr_scheduler')
        self.register_best_state('model')
        self.autocast_dtype = {
            'float16': torch.float16, 'bfloat16': torch.bfloat16
        }[self.cfg.autocast_dtype]
        self.scaler: tp.Optional[torch.cuda.amp.GradScaler] = None # type: ignore
        if self.cfg.fsdp.use:
            need_scaler = self.cfg.fsdp.param_dtype == 'float16'
        else:
            need_scaler = self.cfg.autocast and self.autocast_dtype is torch.float16
        if need_scaler:
            if self.cfg.fsdp.use:
                from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
                self.scaler = ShardedGradScaler()  # type: ignore
            else:
                self.scaler = torch.cuda.amp.GradScaler() # type: ignore
            self.register_stateful('scaler')

    def build_dataloaders(self) -> None:
        """Instantiate audio dataloaders for each stage."""
        self.dataloaders = builders.get_audio_datasets(self.cfg, dataset_type=self.DATASET_TYPE)

    def show(self) -> None:
        """Show the compression model and LM model."""
        self.logger.info("Compression model:")
        self.log_model_summary(self.compression_model)
        self.logger.info("LM model:")
        self.log_model_summary(self.model)

    def load_state_dict(self, state: dict) -> None:
        if 'condition_provider' in state:
            model_state = state['model']
            condition_provider_state = state.pop('condition_provider')
            prefix = 'condition_provider.'
            for key, value in condition_provider_state.items():
                key = prefix + key
                assert key not in model_state
                model_state[key] = value
        if 'compression_model' in state:
            # We used to store the `compression_model` state in the checkpoint, however
            # this is in general not needed, as the compression model should always be readable
            # from the original `cfg.compression_model_checkpoint` location.
            compression_model_state = state.pop('compression_model')
            before_hash = model_hash(self.compression_model)
            self.compression_model.load_state_dict(compression_model_state)
            after_hash = model_hash(self.compression_model)
            if before_hash != after_hash:
                raise RuntimeError(
                    "The compression model state inside the checkpoint is different"
                    " from the one obtained from compression_model_checkpoint..."
                    "We do not support altering the compression model inside the LM "
                    "checkpoint as parts of the code, in particular for running eval post-training "
                    "will use the compression_model_checkpoint as the source of truth.")

        super().load_state_dict(state)

    def load_from_pretrained(self, name: str):
        # TODO: support native HF versions of MusicGen.
        lm_pkg = models.loaders.load_lm_model_ckpt(name)
        state: dict = {
            'best_state': {
                'model': lm_pkg['best_state'],
            },
        }
        return state

    # based on https://github.com/eric-mitchell/direct-preference-optimization/blob/8a4023f5a5bde957b5d44f687569e075ff54e4f7/trainers.py#L45
    def _compute_dpo_loss(self,
                          yw_ce: torch.Tensor,
                          yl_ce: torch.Tensor,
                          yw_ref_ce: torch.Tensor,
                          yl_ref_ce: torch.Tensor,
                          beta_p: tp.Optional[float] = None,
                          loss_type_p: tp.Optional[tp.Union[tp.Literal["sigmoid", "robust", "ipo"], str]] = None,
                          label_smoothing_p: tp.Optional[float] = None,
                          ) -> torch.Tensor:
        beta: float = self.cfg.optim.dpo.beta if beta_p is None else beta_p
        loss_type: tp.Union[tp.Literal["sigmoid", "robust", "ipo"], str] = self.cfg.optim.dpo.loss_type if loss_type_p is None else loss_type_p
        label_smoothing: float = self.cfg.optim.dpo.label_smoothing if label_smoothing_p is None else label_smoothing_p

        pi_yw_logps, pi_yl_logps = -yw_ce, -yl_ce
        ref_yw_logps, ref_yl_logps = -yw_ref_ce, -yl_ref_ce

        pi_logratios = pi_yw_logps - pi_yl_logps
        ref_logratios = ref_yw_logps - ref_yl_logps

        logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

        if loss_type == "ipo":
            losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        elif loss_type == "robust":
            losses = ( -F.logsigmoid(beta * logits) * (1 - label_smoothing) + F.logsigmoid(-beta * logits) * label_smoothing ) / (1 - 2 * label_smoothing) # via https://github.com/huggingface/trl/blob/cbcaa46cd3c02c0e7f724b764c5848ae73796de7/trl/trainer/dpo_trainer.py#L1119
        elif loss_type == "sigmoid":
            losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # chosen_rewards = beta * (pi_yw_logps - ref_yw_logps).detach()
        # rejected_rewards = beta * (pi_yl_logps - ref_yl_logps).detach()

        # print(f"losses.shape: {losses.shape}, torch.mean(losses): {torch.mean(losses)}, pi_logratios.shape: {pi_logratios.shape}, torch.mean(pi_logratios): {torch.mean(pi_logratios)}, torch.mean(ref_logratios): {torch.mean(ref_logratios)}, delta: {torch.mean(pi_logratios) - torch.mean(ref_logratios)}")

        return losses #, chosen_rewards, rejected_rewards

    @dataclass
    class _CEResults:
        ce: torch.Tensor # [B]
        ce_wavg: torch.Tensor # []
        ce_wavg_per_codebook: tp.List[torch.Tensor] # [K]

    def _compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, is_negative_bmask: torch.Tensor, bmask: torch.Tensor, negative_weight: float = 1.0
    ) -> _CEResults:
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
            is_negative_bmask (torch.Tensor): Boolean tensor indicating negative examples, of shape [B].
            bmask (torch.Tensor): Boolean tensor indicating specific samples for this cross entropy computation, of shape [B].
            negative_weight (float): Weight for negative samples in the cross entropy computation.
        Returns:
            ce (torch.Tensor): Cross entropy summed over the codebooks
            ce_wavg (torch.Tensor): Weighted average cross entropy over the codebooks
            ce_wavg_per_codebook (list[torch.Tensor]): Weighted average cross entropy per codebook
        """
        targets_pos = targets[bmask & ~is_negative_bmask]
        logits_pos = logits[bmask & ~is_negative_bmask]
        mask_pos = mask[bmask & ~is_negative_bmask]

        targets_neg = targets[bmask & is_negative_bmask]
        logits_neg = logits[bmask & is_negative_bmask]
        mask_neg = mask[bmask & is_negative_bmask]

        B_pos, K_pos, T_pos = targets_pos.shape
        assert logits_pos.shape[:-1] == targets_pos.shape
        assert mask_pos.shape == targets_pos.shape

        B_neg, K_neg, T_neg = targets_neg.shape
        assert logits_neg.shape[:-1] == targets_neg.shape
        assert mask_neg.shape == targets_neg.shape

        assert K_pos == K_neg, f"K: {K_pos}, K_neg: {K_neg}"
        assert T_pos == T_neg, f"T: {T_pos}, T_neg: {T_neg}"
        K = K_pos
        T = T_pos

        ce = torch.zeros(B_pos+B_neg, dtype=torch.float32, device=targets_pos.device)
        ce_wavg = torch.zeros([], device=targets.device)
        ce_wavg_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            q_ce_pos = torch.zeros((B_pos, T), dtype=torch.half, device=targets_pos.device)
            q_ce_neg = torch.zeros((B_neg, T), dtype=torch.half, device=targets_pos.device)
            # q_ce = torch.zeros((B_pos+B_neg), dtype=torch.half, device=targets_pos.device) # we could put them back into original order, but it's not necessary, bmask may not be contiguous anyway

            logits_k_pos = logits_pos[:, k, ...] # [B_pos, T, card]
            targets_k_pos = targets_pos[:, k, ...] # [B_pos, T]
            mask_k_pos = mask_pos[:, k, ...] # [B_pos, T]

            logits_k_neg = logits_neg[:, k, ...] # [B_neg, T, card]
            targets_k_neg = targets_neg[:, k, ...] # [B_neg, T]
            mask_k_neg = mask_neg[:, k, ...] # [B_neg, T]

            # print(f"logits_k_pos.shape: {logits_k_pos.shape}, targets_k_pos.shape: {targets_k_pos.shape}, mask_k_pos.shape: {mask_k_pos.shape}, logits_k_neg.shape: {logits_k_neg.shape}, targets_k_neg.shape: {targets_k_neg.shape}, mask_k_neg.shape: {mask_k_neg.shape}")
            ce_targets_pos = targets_k_pos[mask_k_pos]
            ce_logits_pos = logits_k_pos[mask_k_pos]
            ce_targets_neg = targets_k_neg[mask_k_neg]
            ce_logits_neg = logits_k_neg[mask_k_neg]
            # print(f"ce_targets_pos.shape: {ce_targets_pos.shape}, ce_logits_pos.shape: {ce_logits_pos.shape}, ce_targets_neg.shape: {ce_targets_neg.shape}, ce_logits_neg.shape: {ce_logits_neg.shape}")

            # Calculate cross-entropy for positive samples
            if len(ce_targets_pos) > 0:
                q_ce_pos_masked = F.cross_entropy(ce_logits_pos, ce_targets_pos, reduction='none')
                q_ce_pos.masked_scatter_(mask_k_pos, q_ce_pos_masked)

            # Calculate cross-entropy for negative samples
            if len(ce_targets_neg) > 0:
                ce_logits_neg = -ce_logits_neg
                q_ce_neg_masked = F.cross_entropy(ce_logits_neg, ce_targets_neg, reduction='none')
                q_ce_neg.masked_scatter_(mask_k_neg, q_ce_neg_masked)


            q_ce_pos_summed = q_ce_pos.sum(dim=-1) # [B_pos] sum over T
            q_ce_pos_divisor = mask_k_pos.count_nonzero(dim=-1).float() # [B_pos] count valid T
            q_ce_pos_avg = q_ce_pos_summed / q_ce_pos_divisor # [B_pos] average over T

            q_ce_neg_summed = q_ce_neg.sum(dim=-1) # [B_neg] sum over T
            q_ce_neg_divisor = mask_k_neg.count_nonzero(dim=-1).float() # [B_neg] count valid T
            q_ce_neg_avg = q_ce_neg_summed / q_ce_neg_divisor # [B_pos] average over T

            q_ce = torch.cat([q_ce_pos_avg, q_ce_neg_avg], dim=0) # [B_pos+B_neg]

            q_ce_wavg = torch.mean(q_ce_pos_avg).nan_to_num() + negative_weight * torch.mean(q_ce_neg_avg).nan_to_num() # nan_to_num to avoid NaNs if len is 0

            # print(f"q_ce_pos_avg: {q_ce_pos_avg}, q_ce_neg_avg: {q_ce_neg_avg}, q_ce: {q_ce}, q_ce_wavg: {q_ce_wavg}")

            ce += q_ce
            ce_wavg += q_ce_wavg
            ce_wavg_per_codebook.append(q_ce_wavg.detach())
        # average cross entropy across codebooks
        ce = ce / K
        ce_wavg = ce_wavg / K
        return self._CEResults(ce=ce, ce_wavg=ce_wavg, ce_wavg_per_codebook=ce_wavg_per_codebook)

    @dataclass
    class _TokensAndAttributes:
        condition_tensors: dict
        audio_tokens: torch.Tensor
        padding_mask: torch.Tensor

    def _prepare_tokens_and_attributes(
        self, batch: tp.Tuple[torch.Tensor, tp.Sequence[SegmentWithAttributes]],
        check_synchronization_points: bool = False
    ) -> _TokensAndAttributes:
        """Prepare input batchs for language model training.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]): Input batch with audio tensor of shape [B, C, T]
                and corresponding metadata as SegmentWithAttributes (with B items).
            check_synchronization_points (bool): Whether to check for synchronization points slowing down training.
        Returns:
            Condition tensors (dict[str, any]): Preprocessed condition attributes.
            Tokens (torch.Tensor): Audio tokens from compression model, of shape [B, K, T_s],
                with B the batch size, K the number of codebooks, T_s the token timesteps.
            Padding mask (torch.Tensor): Mask with valid positions in the tokens tensor, of shape [B, K, T_s].
        """
        if self.model.training:
            warnings.warn(
                "Up to version 1.0.1, the _prepare_tokens_and_attributes was evaluated with `torch.no_grad()`. "
                "This is inconsistent with how model were trained in the MusicGen paper. We removed the "
                "`torch.no_grad()` in version 1.1.0. Small changes to the final performance are expected. "
                "Really sorry about that.")
        if self._cached_batch_loader is None or self.current_stage != "train":
            audio, infos = batch
            audio = audio.to(self.device)
            audio_tokens = None
            assert audio.size(0) == len(infos), (
                f"Mismatch between number of items in audio batch ({audio.size(0)})",
                f" and in metadata ({len(infos)})"
            )
        else:
            audio = None
            # In that case the batch will be a tuple coming from the _cached_batch_writer bit below.
            infos, = batch  # type: ignore
            assert all([isinstance(info, AudioInfo) for info in infos])
            assert all([info.audio_tokens is not None for info in infos])  # type: ignore
            audio_tokens = torch.stack([info.audio_tokens for info in infos]).to(self.device)  # type: ignore
            audio_tokens = audio_tokens.long()
            for info in infos:
                if isinstance(info, MusicInfo):
                    # Careful here, if you want to use this condition_wav (e.b. chroma conditioning),
                    # then you must be using the chroma cache! otherwise the code will try
                    # to use this segment and fail (by that I mean you will see NaN everywhere).
                    info.self_wav = WavCondition(
                        torch.full([1, info.channels, info.total_frames], float('NaN')),
                        length=torch.tensor([info.n_frames]),
                        sample_rate=[info.sample_rate],
                        path=[info.meta.path],
                        seek_time=[info.seek_time])
                    dataset = get_dataset_from_loader(self.dataloaders['original_train'])
                    assert isinstance(dataset, MusicDataset), type(dataset)
                    if dataset.paraphraser is not None and info.description is not None:
                        # Hackingly reapplying paraphraser when using cache.
                        info.description = dataset.paraphraser.sample_paraphrase(
                            info.meta.path, info.description)
        # prepare attributes
        attributes = [info.to_condition_attributes() for info in infos] # type: ignore
        attributes = self.model.cfg_dropout(attributes)
        attributes = self.model.att_dropout(attributes)
        tokenized = self.model.condition_provider.tokenize(attributes)

        # Now we should be synchronization free.
        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("warn")

        if audio_tokens is None:
            with torch.no_grad():
                audio_tokens, scale = self.compression_model.encode(audio) # type: ignore
                assert scale is None, "Scaled compression model not supported with LM."

        with self.autocast:
            condition_tensors = self.model.condition_provider(tokenized)

        # create a padding mask to hold valid vs invalid positions
        padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool, device=audio_tokens.device)
        # replace encodec tokens from padded audio with special_token_id
        if self.cfg.tokens.padding_with_special_token:
            audio_tokens = audio_tokens.clone()
            padding_mask = padding_mask.clone()
            token_sample_rate = self.compression_model.frame_rate
            B, K, T_s = audio_tokens.shape
            for i in range(B):
                n_samples = infos[i].n_frames # type: ignore
                audio_sample_rate = infos[i].sample_rate # type: ignore
                # take the last token generated from actual audio frames (non-padded audio)
                valid_tokens = math.floor(float(n_samples) / audio_sample_rate * token_sample_rate)
                audio_tokens[i, :, valid_tokens:] = self.model.special_token_id
                padding_mask[i, :, valid_tokens:] = 0

        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("default")

        if self._cached_batch_writer is not None and self.current_stage == 'train':
            assert self._cached_batch_loader is None
            assert audio_tokens is not None
            for info, one_audio_tokens in zip(infos, audio_tokens):
                assert isinstance(info, AudioInfo)
                if isinstance(info, MusicInfo):
                    assert not info.joint_embed, "joint_embed and cache not supported yet."
                    info.self_wav = None
                assert one_audio_tokens.max() < 2**15, one_audio_tokens.max().item()
                info.audio_tokens = one_audio_tokens.short().cpu()
            self._cached_batch_writer.save(infos)

        return self._TokensAndAttributes(condition_tensors=condition_tensors, audio_tokens=audio_tokens, padding_mask=padding_mask)

    @dataclass
    class _DPOBatchSplits:
        new_batch: tp.Tuple[torch.Tensor, tp.Sequence[SegmentWithAttributes]]
        dpo_yl_bmask: torch.Tensor
        dpo_yw_bmask: torch.Tensor
        nondpo_bmask: torch.Tensor
        is_negative_bmask: torch.Tensor
        dpo_any: bool
        nondpo_any: bool


    def _create_dpo_batch_splits(self, infos: tp.List[SoundInfo]) -> _DPOBatchSplits:
        assert self.current_dataloader is not None
        if isinstance(self.current_dataloader.dataset, SoundDataset):
            curr_dataset = self.current_dataloader.dataset
        elif isinstance(self.current_dataloader.dataset.dataset, SoundDataset): # if self.current_dataloader.dataset is torch.utils.data.dataset.Subset
            curr_dataset = self.current_dataloader.dataset.dataset
        else:
            raise RuntimeError(f"Could not find SoundDataset in current_dataloader.dataset, got {type(self.current_dataloader.dataset)}")

        dpo_l_list, dpo_w_list, nondpo_list = [], [], []
        for info in infos:
            has_dpo = bool(hasattr(info, "self_l_wav") and info.self_l_wav is not None)
            if has_dpo:
                assert has_dpo and not info.negative, "If sample's meta supports DPO, it should be the positive example."
                assert info.self_wav is not None and info.self_l_wav is not None
                dpo_l_list.append((info.self_l_wav.wav[0], info))
                dpo_w_list.append((info.self_wav.wav[0], info))
            else:
                assert info.self_wav is not None
                nondpo_list.append((info.self_wav.wav[0], info))

        new_batch_samples = dpo_l_list + dpo_w_list + nondpo_list
        new_batch: tp.Tuple[torch.Tensor, tp.Sequence[SegmentWithAttributes]] = curr_dataset.collater(new_batch_samples, aug_last_n=len(nondpo_list))
        # if augmentation, number of nondpo samples may change
        num_nondpo_samples = len(new_batch[1]) - len(dpo_l_list) - len(dpo_w_list)
        dpo_yl_bmask = torch.tensor([True ] * len(dpo_l_list) + [False] * len(dpo_w_list) + [False] * num_nondpo_samples, device=self.device, dtype=torch.bool)
        dpo_yw_bmask = torch.tensor([False] * len(dpo_l_list) + [True ] * len(dpo_w_list) + [False] * num_nondpo_samples, device=self.device, dtype=torch.bool)
        nondpo_bmask = torch.tensor([False] * len(dpo_l_list) + [False] * len(dpo_w_list) + [True ] * num_nondpo_samples, device=self.device, dtype=torch.bool)

        assert len(dpo_l_list) == len(dpo_w_list)
        dpo_any = len(dpo_l_list) > 0
        nondpo_any = num_nondpo_samples > 0
        # if dpo_any:
        #     assert curr_dataset.aug_p == 0.0, f"DPO may not work well with mixing augmentations set to 0.0, got {curr_dataset.aug_p}" # aug_last_n is set to len(nondpo_list) so this should be fine

        is_negative_bmask = torch.tensor([bool(hasattr(info, "negative") and info.negative) for info in new_batch[1]], device=self.device, dtype=torch.bool)

        return self._DPOBatchSplits(new_batch=new_batch, dpo_yl_bmask=dpo_yl_bmask, dpo_yw_bmask=dpo_yw_bmask, nondpo_bmask=nondpo_bmask, is_negative_bmask=is_negative_bmask, dpo_any=dpo_any, nondpo_any=nondpo_any)

    def _compute_predictions_and_ce_for_bmasks(self,
                                   model: models.LMModel,
                                   audio_tokens: torch.Tensor,
                                   condition_tensors: dict,

                                   padding_mask: torch.Tensor,
                                   is_negative_bmask: torch.Tensor,

                                   ce_bmasks: tp.Sequence[torch.Tensor],
                                   ):
        model_output = model.compute_predictions(audio_tokens, [], condition_tensors)
        logits = model_output.logits
        mask = padding_mask & model_output.mask

        ces_for_bmasks = [self._compute_cross_entropy(logits=logits, targets=audio_tokens, mask=mask,
                                                      is_negative_bmask=is_negative_bmask, bmask=bmask,
                                                      negative_weight=self.cfg.optim.negative_sample_weight) for bmask in ce_bmasks]

        return ces_for_bmasks


    def run_step(self, idx: int, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]], metrics: dict) -> dict: # type: ignore
        """Perform one training or valid step on a given batch."""
        check_synchronization_points = idx == 1 and self.device == 'cuda'

        assert not self.cfg.cache.path and not self._cached_batch_loader and not self._cached_batch_writer # DPO is not tested with batch cache

        assert all([isinstance(info, SoundInfo) for info in batch[1]])
        infos: tp.List[SoundInfo] = batch[1] # type: ignore asserted above

        dpobs = self._create_dpo_batch_splits(infos) # perf todo: check for check_synchronization_points in here too
        new_batch, dpo_yl_bmask, dpo_yw_bmask, nondpo_bmask, is_negative_bmask, dpo_any, nondpo_any = dpobs.new_batch, dpobs.dpo_yl_bmask, dpobs.dpo_yw_bmask, dpobs.nondpo_bmask, dpobs.is_negative_bmask, dpobs.dpo_any, dpobs.nondpo_any
        nb_ta = self._prepare_tokens_and_attributes(new_batch, check_synchronization_points)

        self.deadlock_detect.update('tokens_and_conditions')

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode('warn')

        with self.autocast:
            dpo_yl_ce_tuple, dpo_yw_ce_tuple, nondpo_ce_tuple = self._compute_predictions_and_ce_for_bmasks(self.model, nb_ta.audio_tokens, nb_ta.condition_tensors, nb_ta.padding_mask, is_negative_bmask, [dpo_yl_bmask, dpo_yw_bmask, nondpo_bmask])
            stats_ce_tuples: tp.List[MusicGenSolver._CEResults] = []
            if dpo_any:
                assert self.reference_model is not None, "DPO samples require a reference model"
                stats_ce_tuples.append(dpo_yw_ce_tuple)

                # perf: could potentially skip nondpo for reference model? need to adjust all inputs for new batch size
                dpo_yl_ref_ce_tuple, dpo_yw_ref_ce_tuple, nondpo_ref_ce_tuple = self._compute_predictions_and_ce_for_bmasks(self.reference_model, nb_ta.audio_tokens, nb_ta.condition_tensors, nb_ta.padding_mask, is_negative_bmask, [dpo_yl_bmask, dpo_yw_bmask, nondpo_bmask])

                dpo_losses = self._compute_dpo_loss(
                    yw_ce=dpo_yw_ce_tuple.ce,
                    yl_ce=dpo_yl_ce_tuple.ce,
                    yw_ref_ce=dpo_yw_ref_ce_tuple.ce,
                    yl_ref_ce=dpo_yl_ref_ce_tuple.ce,
                )
                dpo_loss = torch.mean(dpo_losses) + self.cfg.optim.dpo.ce_mix_beta * torch.mean(dpo_yw_ce_tuple.ce) # we dont include the YL CE because it is calculated using the normal CE calc instead of the flipped logits calc used for negative samples.
            else:
                dpo_loss = None

            if nondpo_any:
                stats_ce_tuples.append(nondpo_ce_tuple)
                nondpo_ce_loss = nondpo_ce_tuple.ce_wavg
            else:
                nondpo_ce_loss = None

            # for metrics:
            ce_for_metrics = torch.mean(torch.stack([cer.ce_wavg for cer in stats_ce_tuples]))
            ce_per_codebook_for_metrics = [torch.mean(torch.stack([cer.ce_wavg_per_codebook[k] for cer in stats_ce_tuples])) for k in range(self.compression_model.num_codebooks)]
            if dpo_loss is not None:
                dpo_loss_for_metrics = dpo_loss
            else:
                dpo_loss_for_metrics = torch.zeros((), device=self.device)

            if dpo_loss is not None and nondpo_ce_loss is not None:
                loss = dpo_loss + nondpo_ce_loss
            elif dpo_loss is not None and nondpo_ce_loss is None:
                loss = dpo_loss
            elif dpo_loss is None and nondpo_ce_loss is not None:
                loss = nondpo_ce_loss
            else:
                raise RuntimeError("No loss computable")

        self.deadlock_detect.update('loss')

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode('default')

        if self.is_training:
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            if self.scaler is not None:
                loss: torch.Tensor = self.scaler.scale(loss) # type: ignore
            self.deadlock_detect.update('scale')
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
            self.deadlock_detect.update('backward')

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.cfg.optim.max_norm:
                if self.cfg.fsdp.use:
                    metrics['grad_norm'] = self.model.clip_grad_norm_(self.cfg.optim.max_norm)  # type: ignore
                else:
                    metrics['grad_norm'] = torch.nn.utils.clip_grad_norm_( # type: ignore
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
            self.deadlock_detect.update('optim')
            if self.scaler is not None:
                scale = self.scaler.get_scale()
                metrics['grad_scale'] = scale
            if not loss.isfinite().all():
                raise RuntimeError("Model probably diverged.")


        ce = ce_for_metrics
        ce_per_codebook = ce_per_codebook_for_metrics
        metrics['ce'] = ce
        metrics['ppl'] = torch.exp(ce)
        for k, ce_q in enumerate(ce_per_codebook):
            metrics[f'ce_q{k + 1}'] = ce_q
            metrics[f'ppl_q{k + 1}'] = torch.exp(ce_q)
        metrics['dpo_loss'] = dpo_loss_for_metrics

        return metrics

    @torch.no_grad()
    def run_generate_step(self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
                          gen_duration: float, prompt_duration: tp.Optional[float] = None,
                          remove_prompt: bool = False,
                          **generation_params) -> dict:
        """Run generate step on a batch of optional audio tensor and corresponding attributes.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]):
            use_prompt (bool): Whether to do audio continuation generation with prompt from audio batch.
            gen_duration (float): Target audio duration for the generation.
            prompt_duration (float, optional): Duration for the audio prompt to use for continuation.
            remove_prompt (bool, optional): Whether to remove the prompt from the generated audio.
            generation_params: Additional generation parameters.
        Returns:
            gen_outputs (dict): Generation outputs, consisting in audio, audio tokens from both the generation
                and the prompt along with additional information.
        """
        bench_start = time.time()
        audio, meta = batch
        assert audio.size(0) == len(meta), (
            f"Mismatch between number of items in audio batch ({audio.size(0)})",
            f" and in metadata ({len(meta)})"
        )
        # prepare attributes
        attributes = [x.to_condition_attributes() for x in meta]
        # TODO: Add dropout for chroma?

        # prepare audio prompt
        if prompt_duration is None:
            prompt_audio = None
        else:
            assert prompt_duration < gen_duration, "Prompt duration must be lower than target generation duration"
            prompt_audio_frames = int(prompt_duration * self.compression_model.sample_rate)
            prompt_audio = audio[..., :prompt_audio_frames]

        # get audio tokens from compression model
        if prompt_audio is None or prompt_audio.nelement() == 0:
            num_samples = len(attributes)
            prompt_tokens = None
        else:
            num_samples = None
            prompt_audio = prompt_audio.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt_audio)
            assert scale is None, "Compression model in MusicGen should not require rescaling."

        # generate by sampling from the LM
        with self.autocast:
            total_gen_len = math.ceil(gen_duration * self.compression_model.frame_rate)
            gen_tokens = self.model.generate(
                prompt_tokens, attributes, max_gen_len=total_gen_len,
                num_samples=num_samples, **self.generation_params)

        # generate audio from tokens
        assert gen_tokens.dim() == 3
        gen_audio = self.compression_model.decode(gen_tokens, None)

        bench_end = time.time()
        gen_outputs = {
            'rtf': (bench_end - bench_start) / gen_duration,
            'ref_audio': audio,
            'gen_audio': gen_audio,
            'gen_tokens': gen_tokens,
            'prompt_audio': prompt_audio,
            'prompt_tokens': prompt_tokens,
        }
        return gen_outputs

    def generate_audio(self) -> dict:
        """Audio generation stage."""
        generate_stage_name = f'{self.current_stage}'
        sample_manager = SampleManager(self.xp)
        self.logger.info(f"Generating samples in {sample_manager.base_folder}")
        loader = self.dataloaders['generate']
        updates = len(loader)
        lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

        dataset = get_dataset_from_loader(loader)
        dataset_duration = dataset.segment_duration
        assert dataset_duration is not None
        assert isinstance(dataset, AudioDataset)
        target_duration = self.cfg.generate.lm.gen_duration
        prompt_duration = self.cfg.generate.lm.prompt_duration
        if target_duration is None:
            target_duration = dataset_duration
        if prompt_duration is None:
            prompt_duration = dataset_duration / 4
        assert prompt_duration < dataset_duration, (
            f"Specified prompt duration ({prompt_duration}s) is longer",
            f" than reference audio duration ({dataset_duration}s)"
        )

        def get_hydrated_conditions(meta: tp.List[SegmentWithAttributes]):
            hydrated_conditions = []
            for sample in [x.to_condition_attributes() for x in meta]:
                cond_dict = {}
                for cond_type in sample.__annotations__.keys():
                    for cond_key, cond_val in getattr(sample, cond_type).items():
                        if cond_key not in self.model.condition_provider.conditioners.keys():
                            continue
                        if is_jsonable(cond_val):
                            cond_dict[cond_key] = cond_val
                        elif isinstance(cond_val, WavCondition):
                            cond_dict[cond_key] = cond_val.path
                        elif isinstance(cond_val, JointEmbedCondition):
                            cond_dict[cond_key] = cond_val.text  # only support text at inference for now
                        else:
                            # if we reached this point, it is not clear how to log the condition
                            # so we just log the type.
                            cond_dict[cond_key] = str(type(cond_val))
                            continue
                hydrated_conditions.append(cond_dict)
            return hydrated_conditions

        metrics: dict = {}
        average = flashy.averager() # type: ignore
        for batch in lp:
            audio, meta = batch
            # metadata for sample manager
            hydrated_conditions = get_hydrated_conditions(meta)
            sample_generation_params = {
                **{f'classifier_free_guidance_{k}': v for k, v in self.cfg.classifier_free_guidance.items()},
                **self.generation_params
            }
            if self.cfg.generate.lm.unprompted_samples:
                if self.cfg.generate.lm.gen_gt_samples:
                    # get the ground truth instead of generation
                    self.logger.warn(
                        "Use ground truth instead of audio generation as generate.lm.gen_gt_samples=true")
                    gen_unprompted_audio = audio
                    rtf = 1.
                else:
                    gen_unprompted_outputs = self.run_generate_step(
                        batch, gen_duration=target_duration, prompt_duration=None,
                        **self.generation_params)
                    gen_unprompted_audio = gen_unprompted_outputs['gen_audio'].cpu()
                    rtf = gen_unprompted_outputs['rtf']
                sample_manager.add_samples(
                    gen_unprompted_audio, self.epoch, hydrated_conditions,
                    ground_truth_wavs=audio, generation_args=sample_generation_params)

            if self.cfg.generate.lm.prompted_samples:
                gen_outputs = self.run_generate_step(
                    batch, gen_duration=target_duration, prompt_duration=prompt_duration,
                    **self.generation_params)
                gen_audio = gen_outputs['gen_audio'].cpu()
                prompt_audio = gen_outputs['prompt_audio'].cpu()
                sample_manager.add_samples(
                    gen_audio, self.epoch, hydrated_conditions,
                    prompt_wavs=prompt_audio, ground_truth_wavs=audio,
                    generation_args=sample_generation_params)

            metrics['rtf'] = rtf # type: ignore
            metrics = average(metrics)

        flashy.distrib.barrier()
        return metrics

    def generate(self) -> dict: # type: ignore
        """Generate stage."""
        self.model.eval()
        with torch.no_grad():
            return self.generate_audio()

    def run_epoch(self):
        if self.cfg.cache.write:
            if ((self.epoch - 1) % self.cfg.cache.write_num_shards) != self.cfg.cache.write_shard:
                return
        super().run_epoch()

    def train(self):
        """Train stage.
        """
        if self._cached_batch_writer is not None:
            self._cached_batch_writer.start_epoch(self.epoch)
        if self._cached_batch_loader is None:
            dataset = get_dataset_from_loader(self.dataloaders['train'])
            assert isinstance(dataset, AudioDataset)
            dataset.current_epoch = self.epoch
        else:
            self._cached_batch_loader.start_epoch(self.epoch)
        return super().train()

    def evaluate_audio_generation(self) -> dict:
        """Evaluate audio generation with off-the-shelf metrics."""
        evaluate_stage_name = f'{self.current_stage}_generation'
        # instantiate evaluation metrics, if at least one metric is defined, run audio generation evaluation
        fad: tp.Optional[eval_metrics.FrechetAudioDistanceMetric] = None
        kldiv: tp.Optional[eval_metrics.KLDivergenceMetric] = None
        text_consistency: tp.Optional[eval_metrics.TextConsistencyMetric] = None
        chroma_cosine: tp.Optional[eval_metrics.ChromaCosineSimilarityMetric] = None
        should_run_eval = False
        eval_chroma_wavs: tp.Optional[torch.Tensor] = None
        if self.cfg.evaluate.metrics.fad:
            fad = builders.get_fad(self.cfg.metrics.fad).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.kld:
            kldiv = builders.get_kldiv(self.cfg.metrics.kld).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.text_consistency:
            text_consistency = builders.get_text_consistency(self.cfg.metrics.text_consistency).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.chroma_cosine:
            chroma_cosine = builders.get_chroma_cosine_similarity(self.cfg.metrics.chroma_cosine).to(self.device)
            # if we have predefind wavs for chroma we should purge them for computing the cosine metric
            has_predefined_eval_chromas = 'self_wav' in self.model.condition_provider.conditioners and \
                                          self.model.condition_provider.conditioners['self_wav'].has_eval_wavs()
            if has_predefined_eval_chromas:
                warn_once(self.logger, "Attempting to run cosine eval for config with pre-defined eval chromas! "
                                       'Resetting eval chromas to None for evaluation.')
                eval_chroma_wavs = self.model.condition_provider.conditioners.self_wav.eval_wavs  # type: ignore
                self.model.condition_provider.conditioners.self_wav.reset_eval_wavs(None)  # type: ignore
            should_run_eval = True

        def get_compressed_audio(audio: torch.Tensor) -> torch.Tensor:
            audio_tokens, scale = self.compression_model.encode(audio.to(self.device))
            compressed_audio = self.compression_model.decode(audio_tokens, scale)
            return compressed_audio[..., :audio.shape[-1]] # type: ignore

        metrics: dict = {}
        if should_run_eval:
            loader = self.dataloaders['evaluate']
            updates = len(loader)
            lp = self.log_progress(f'{evaluate_stage_name} inference', loader, total=updates, updates=self.log_updates)
            average = flashy.averager() # type: ignore
            dataset = get_dataset_from_loader(loader)
            assert isinstance(dataset, AudioDataset)
            self.logger.info(f"Computing evaluation metrics on {len(dataset)} samples")

            for idx, batch in enumerate(lp):
                audio, meta = batch
                assert all([self.cfg.sample_rate == m.sample_rate for m in meta])

                target_duration = audio.shape[-1] / self.cfg.sample_rate
                if self.cfg.evaluate.fixed_generation_duration:
                    target_duration = self.cfg.evaluate.fixed_generation_duration

                gen_outputs = self.run_generate_step(
                    batch, gen_duration=target_duration,
                    **self.generation_params
                )
                y_pred = gen_outputs['gen_audio'].detach()
                y_pred = y_pred[..., :audio.shape[-1]]

                normalize_kwargs = dict(self.cfg.generate.audio)
                normalize_kwargs.pop('format', None)
                y_pred = torch.stack([normalize_audio(w, **normalize_kwargs) for w in y_pred], dim=0).cpu()
                y = audio.cpu()  # should already be on CPU but just in case
                sizes = torch.tensor([m.n_frames for m in meta])  # actual sizes without padding
                sample_rates = torch.tensor([m.sample_rate for m in meta])  # sample rates for audio samples
                audio_stems = [Path(m.meta.path).stem + f"_{m.seek_time}" for m in meta]

                if fad is not None:
                    if self.cfg.metrics.fad.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    fad.update(y_pred, y, sizes, sample_rates, audio_stems)
                if kldiv is not None:
                    if self.cfg.metrics.kld.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    kldiv.update(y_pred, y, sizes, sample_rates)
                if text_consistency is not None:
                    texts = [m.description for m in meta]
                    if self.cfg.metrics.text_consistency.use_gt:
                        y_pred = y
                    text_consistency.update(y_pred, texts, sizes, sample_rates)
                if chroma_cosine is not None:
                    if self.cfg.metrics.chroma_cosine.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    chroma_cosine.update(y_pred, y, sizes, sample_rates)
                    # restore chroma conditioner's eval chroma wavs
                    if eval_chroma_wavs is not None:
                        self.model.condition_provider.conditioners['self_wav'].reset_eval_wavs(eval_chroma_wavs)

            flashy.distrib.barrier()
            if fad is not None:
                metrics['fad'] = fad.compute()
            if kldiv is not None:
                kld_metrics = kldiv.compute()
                metrics.update(kld_metrics)
            if text_consistency is not None:
                metrics['text_consistency'] = text_consistency.compute()
            if chroma_cosine is not None:
                metrics['chroma_cosine'] = chroma_cosine.compute()
            metrics = average(metrics)
            metrics = flashy.distrib.average_metrics(metrics, len(loader))

        return metrics

    def evaluate(self) -> dict: # type: ignore
        """Evaluate stage."""
        self.model.eval()
        with torch.no_grad():
            metrics: dict = {}
            if self.cfg.evaluate.metrics.base:
                metrics.update(self.common_train_valid('evaluate'))
            gen_metrics = self.evaluate_audio_generation()
            return {**metrics, **gen_metrics}
