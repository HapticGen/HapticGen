# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Dataset of audio with a simple description.
"""

import copy
from dataclasses import dataclass, fields, replace
import json
import logging
from pathlib import Path
import random
import typing as tp

from audiocraft.data.audio import audio_read
from audiocraft.data.audio_dataset import AudioMeta, SegmentInfo
from audiocraft.data.audio_utils import convert_audio


logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn.functional as F

from .info_audio_dataset import (
    InfoAudioDataset,
    get_keyword_or_keyword_list
)
from ..modules.conditioners import (
    ConditioningAttributes,
    SegmentWithAttributes,
    WavCondition,
)


EPS = torch.finfo(torch.float32).eps
TARGET_LEVEL_LOWER = -35
TARGET_LEVEL_UPPER = -15


@dataclass
class SoundInfo(SegmentWithAttributes):
    """Segment info augmented with Sound metadata.
    """
    description: tp.Optional[str] = None
    self_wav: tp.Optional[WavCondition] = None
    self_l_wav: tp.Optional[WavCondition] = None
    l_wav_path: tp.Optional[str] = None
    negative: tp.Optional[bool] = None

    @property
    def has_sound_meta(self) -> bool:
        return self.description is not None

    def to_condition_attributes(self) -> ConditioningAttributes:
        out = ConditioningAttributes()

        for _field in fields(self):
            key, value = _field.name, getattr(self, _field.name)
            if key == 'self_wav':
                out.wav[key] = value
            elif key == 'description':
                out.text[key] = value
            else:
                pass # ignore other fields
        return out

    @staticmethod
    def attribute_getter(attribute):
        if attribute == 'description':
            preprocess_func = get_keyword_or_keyword_list
        else:
            preprocess_func = None
        return preprocess_func

    @classmethod
    def from_dict(cls, dictionary: dict, fields_required: bool = False):
        _dictionary: tp.Dict[str, tp.Any] = {}

        # allow a subset of attributes to not be loaded from the dictionary
        # these attributes may be populated later
        post_init_attributes = ['self_wav', 'joint_embed', 'self_l_wav'] # joint_embed is not used in SoundDataset, but here anyway
        always_optional = ['negative', 'l_wav_path']

        for _field in fields(cls):
            if _field.name in post_init_attributes:
                continue
            elif _field.name not in dictionary:
                if fields_required and _field.name not in always_optional:
                    raise KeyError(f"Unexpected missing key: {_field.name}")
            else:
                preprocess_func: tp.Optional[tp.Callable] = cls.attribute_getter(_field.name)
                value = dictionary[_field.name]
                if preprocess_func:
                    value = preprocess_func(value)
                _dictionary[_field.name] = value
        return cls(**_dictionary)


class SoundDataset(InfoAudioDataset):
    """Sound audio dataset: Audio dataset with environmental sound-specific metadata.

    Args:
        info_fields_required (bool): Whether all the mandatory metadata fields should be in the loaded metadata.
        external_metadata_source (tp.Optional[str]): Folder containing JSON metadata for the corresponding dataset.
            The metadata files contained in this folder are expected to match the stem of the audio file with
            a json extension.
        metadata_suffixes (str): Suffixes for metadata files. Default is ['.json']. (ex: ".fullcmb.json", ".captiononly.json")
        aug_p (float): Probability of performing audio mixing augmentation on the batch.
        mix_p (float): Proportion of batch items that are mixed together when applying audio mixing augmentation.
        mix_snr_low (int): Lowerbound for SNR value sampled for mixing augmentation.
        mix_snr_high (int): Upperbound for SNR value sampled for mixing augmentation.
        mix_min_overlap (float): Minimum overlap between audio files when performing mixing augmentation.
        kwargs: Additional arguments for AudioDataset.

    See `audiocraft.data.info_audio_dataset.InfoAudioDataset` for full initialization arguments.
    """
    def __init__(
        self,
        *args,
        info_fields_required: bool = True,
        external_metadata_source: tp.Optional[str] = None,
        metadata_suffixes: list[str] = ['.json'],
        aug_p: float = 0.,
        mix_p: float = 0.,
        mix_snr_low: int = -5,
        mix_snr_high: int = 5,
        mix_min_overlap: float = 0.5,
        **kwargs
    ):
        kwargs['return_info'] = True  # We require the info for each song of the dataset.
        super().__init__(*args, **kwargs)
        self.info_fields_required = info_fields_required
        self.external_metadata_source = external_metadata_source
        self.metadata_suffixes = metadata_suffixes
        self.aug_p = aug_p
        self.mix_p = mix_p
        if self.aug_p > 0:
            assert self.mix_p > 0, "Expecting some mixing proportion mix_p if aug_p > 0"
            assert self.channels == 1, "SoundDataset with audio mixing considers only monophonic audio"
        self.mix_snr_low = mix_snr_low
        self.mix_snr_high = mix_snr_high
        self.mix_min_overlap = mix_min_overlap

    def _get_info_path(self, path: tp.Union[str, Path]) -> tp.Optional[Path]:
        """Get path of JSON with metadata (description, etc.).
        If there exists a JSON with the same name as 'path.name', then it will be used.
        Else, such JSON will be searched for in an external json source folder if it exists.
        """
        for suffix in self.metadata_suffixes:
            info_path = Path(path).with_suffix(suffix)
            if Path(info_path).exists():
                return info_path
            elif self.external_metadata_source and (Path(self.external_metadata_source) / info_path.name).exists():
                return Path(self.external_metadata_source) / info_path.name
        else:
        #     raise Exception(f"Unable to find a metadata JSON for path: {path}")
            return None

    def __getitem__(self, index):
        info: SegmentWithAttributes
        wav, info = super().__getitem__(index) # type: ignore ( info is always SegmentWithAttributes if self.return_info is True)
        info_data = info.to_dict()
        info_path = self._get_info_path(info.meta.path)
        if info_path is not None and Path(info_path).exists():
            try:
                with open(info_path, 'r') as json_file:
                    sound_data = json.load(json_file)
                    sound_data.update(info_data)
                    sound_info = SoundInfo.from_dict(sound_data, fields_required=self.info_fields_required)
                    # if there are multiple descriptions, sample one randomly
                    if isinstance(sound_info.description, list):
                        sound_info.description = random.choice(sound_info.description)
                    if isinstance(sound_info.l_wav_path, list): # if there are multiple l_wav_paths, sample one randomly
                        sound_info.l_wav_path = random.choice(sound_info.l_wav_path)
            except Exception as e:
                logger.error(f"Error loading metadata from {info_path}")
                e.args += (f"Error loading metadata from {info_path}",)
                raise
        else:
            sound_info = SoundInfo.from_dict(info_data, fields_required=False)


        lwav_meta = copy.deepcopy(sound_info.meta) # must copy original meta (must not be negative=true)
        if sound_info.l_wav_path is not None:
            lwav_meta.path = sound_info.l_wav_path
            lwav, lwav_info = self.get_preference_l_wav(index, lwav_meta)
            sound_info.self_l_wav = WavCondition(
                wav=lwav[None], length=torch.tensor([lwav_info.n_frames]),
                sample_rate=[sound_info.sample_rate], path=[lwav_meta.path], seek_time=[lwav_info.seek_time])

        sound_info.self_wav = WavCondition(
            wav=wav[None], length=torch.tensor([info.n_frames]),
            sample_rate=[sound_info.sample_rate], path=[info.meta.path], seek_time=[info.seek_time])

        return wav, sound_info

    def get_preference_l_wav(self, index, file_meta: AudioMeta) -> tp.Tuple[torch.Tensor, SegmentInfo]:
        if self.segment_duration is None:
            out, sr = audio_read(file_meta.path)
            out = convert_audio(out, sr, self.sample_rate, self.channels)
            n_frames = out.shape[-1]
            segment_info = SegmentInfo(file_meta, seek_time=0., n_frames=n_frames, total_frames=n_frames,
                                       sample_rate=self.sample_rate, channels=out.shape[0])
        else:
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

            assert self.max_read_retry > 0, "max_read_retry should be at least 1"
            for retry in range(self.max_read_retry):
                # We add some variance in the file position even if audio file is smaller than segment
                # without ending up with empty segments
                max_seek = max(0, file_meta.duration - self.segment_duration * self.min_segment_ratio)
                seek_time = torch.rand(1, generator=rng).item() * max_seek
                try:
                    out, sr = audio_read(file_meta.path, seek_time, self.segment_duration, pad=False)
                    out = convert_audio(out, sr, self.sample_rate, self.channels)
                    n_frames = out.shape[-1]
                    target_frames = int(self.segment_duration * self.sample_rate)
                    if self.pad:
                        out = F.pad(out, (0, target_frames - n_frames))
                    segment_info = SegmentInfo(file_meta, seek_time, n_frames=n_frames, total_frames=target_frames,
                                               sample_rate=self.sample_rate, channels=out.shape[0])
                except Exception as exc:
                    logger.warning("Error opening file %s: %r", file_meta.path, exc)
                    if retry == self.max_read_retry - 1:
                        raise
                else:
                    break

        return out, segment_info # type: ignore  max_read_retry always at least 1

    def collater(self, samples, aug_last_n: tp.Optional[int] = None):
        # when training, audio mixing is performed in the collate function
        sound_info: tp.List[SoundInfo]
        wav, sound_info = super().collater(samples) # type: ignore SoundDataset always returns infos
        if self.aug_p > 0:
            if aug_last_n is not None:
                aug_wav = wav[-aug_last_n:]
                aug_sound_info = sound_info[-aug_last_n:]
            else:
                aug_wav = wav
                aug_sound_info = sound_info

            aug_wav, aug_sound_info = mix_samples(aug_wav, aug_sound_info, self.aug_p, self.mix_p,
                                          snr_low=self.mix_snr_low, snr_high=self.mix_snr_high,
                                          min_overlap=self.mix_min_overlap)
            if aug_last_n is not None:
                wav = torch.cat([wav[:-aug_last_n], aug_wav], dim=0)
                sound_info = sound_info[:-aug_last_n] + aug_sound_info
            else:
                wav = aug_wav
                sound_info = aug_sound_info

        return wav, sound_info


def rms_f(x: torch.Tensor) -> torch.Tensor:
    return (x ** 2).mean(1).pow(0.5)


def normalize(audio: torch.Tensor, target_level: int = -25) -> torch.Tensor:
    """Normalize the signal to the target level."""
    rms = rms_f(audio)
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar.unsqueeze(1)
    return audio


def is_clipped(audio: torch.Tensor, clipping_threshold: float = 0.99) -> torch.Tensor:
    return (abs(audio) > clipping_threshold).any(1)


def mix_pair(src: torch.Tensor, dst: torch.Tensor, min_overlap: float) -> torch.Tensor:
    start = random.randint(0, int(src.shape[1] * (1 - min_overlap)))
    remainder = src.shape[1] - start
    if dst.shape[1] > remainder:
        src[:, start:] = src[:, start:] + dst[:, :remainder]
    else:
        src[:, start:start+dst.shape[1]] = src[:, start:start+dst.shape[1]] + dst
    return src


def snr_mixer(clean: torch.Tensor, noise: torch.Tensor, snr: int, min_overlap: float,
              target_level: int = -25, clipping_threshold: float = 0.99) -> torch.Tensor:
    """Function to mix clean speech and noise at various SNR levels.

    Args:
        clean (torch.Tensor): Clean audio source to mix, of shape [B, T].
        noise (torch.Tensor): Noise audio source to mix, of shape [B, T].
        snr (int): SNR level when mixing.
        min_overlap (float): Minimum overlap between the two mixed sources.
        target_level (int): Gain level in dB.
        clipping_threshold (float): Threshold for clipping the audio.
    Returns:
        torch.Tensor: The mixed audio, of shape [B, T].
    """
    if clean.shape[1] > noise.shape[1]:
        noise = torch.nn.functional.pad(noise, (0, clean.shape[1] - noise.shape[1]))
    else:
        noise = noise[:, :clean.shape[1]]

    # normalizing to -25 dB FS
    clean = clean / (clean.max(1)[0].abs().unsqueeze(1) + EPS)
    clean = normalize(clean, target_level)
    rmsclean = rms_f(clean)

    noise = noise / (noise.max(1)[0].abs().unsqueeze(1) + EPS)
    noise = normalize(noise, target_level)
    rmsnoise = rms_f(noise)

    # set the noise level for a given SNR
    noisescalar = (rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)).unsqueeze(1)
    noisenewlevel = noise * noisescalar

    # mix noise and clean speech
    noisyspeech = mix_pair(clean, noisenewlevel, min_overlap)

    # randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # there is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(TARGET_LEVEL_LOWER, TARGET_LEVEL_UPPER)
    rmsnoisy = rms_f(noisyspeech)
    scalarnoisy = (10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)).unsqueeze(1)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy

    # final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    clipped = is_clipped(noisyspeech)
    if clipped.any():
        noisyspeech_maxamplevel = noisyspeech[clipped].max(1)[0].abs().unsqueeze(1) / (clipping_threshold - EPS)
        noisyspeech[clipped] = noisyspeech[clipped] / noisyspeech_maxamplevel

    return noisyspeech


def snr_mix(src: torch.Tensor, dst: torch.Tensor, snr_low: int, snr_high: int, min_overlap: float):
    if snr_low == snr_high:
        snr = snr_low
    else:
        snr = np.random.randint(snr_low, snr_high)
    mix = snr_mixer(src, dst, snr, min_overlap)
    return mix


def mix_text(src_text: tp.Optional[str], dst_text: tp.Optional[str]):
    """Mix text from different sources by concatenating them."""
    if src_text == dst_text:
        return src_text
    if src_text is None:
        return dst_text
    if dst_text is None:
        return src_text
    return src_text + " " + dst_text


def mix_samples(wavs: torch.Tensor, infos: tp.List[SoundInfo], aug_p: float, mix_p: float,
                snr_low: int, snr_high: int, min_overlap: float):
    """Mix samples within a batch, summing the waveforms and concatenating the text infos.

    Args:
        wavs (torch.Tensor): Audio tensors of shape [B, C, T].
        infos (list[SoundInfo]): List of SoundInfo items corresponding to the audio.
        aug_p (float): Augmentation probability.
        mix_p (float): Proportion of items in the batch to mix (and merge) together.
        snr_low (int): Lowerbound for sampling SNR.
        snr_high (int): Upperbound for sampling SNR.
        min_overlap (float): Minimum overlap between mixed samples.
    Returns:
        tuple[torch.Tensor, list[SoundInfo]]: A tuple containing the mixed wavs
            and mixed SoundInfo for the given batch.
    """
    # no mixing to perform within the batch
    if mix_p == 0:
        return wavs, infos

    if random.uniform(0, 1) < aug_p:
        # perform all augmentations on waveforms as [B, T]
        # randomly picking pairs of audio to mix
        assert wavs.size(1) == 1, f"Mix samples requires monophonic audio but C={wavs.size(1)}"
        wavs = wavs.mean(dim=1, keepdim=False)
        B, T = wavs.shape
        k = int(mix_p * B)
        mixed_sources_idx = torch.randperm(B)[:k]
        mixed_targets_idx = torch.randperm(B)[:k]
        aug_wavs = snr_mix(
            wavs[mixed_sources_idx],
            wavs[mixed_targets_idx],
            snr_low,
            snr_high,
            min_overlap,
        )
        # mixing textual descriptions in metadata
        descriptions = [info.description for info in infos]
        aug_infos = []
        for i, j in zip(mixed_sources_idx, mixed_targets_idx):
            text = mix_text(descriptions[i], descriptions[j])
            m = replace(infos[i])
            m.description = text
            aug_infos.append(m)

        # back to [B, C, T]
        aug_wavs = aug_wavs.unsqueeze(1)
        assert aug_wavs.shape[0] > 0, "Samples mixing returned empty batch."
        assert aug_wavs.dim() == 3, f"Returned wav should be [B, C, T] but dim = {aug_wavs.dim()}"
        assert aug_wavs.shape[0] == len(aug_infos), "Mismatch between number of wavs and infos in the batch"

        return aug_wavs, aug_infos  # [B, C, T]
    else:
        # randomly pick samples in the batch to match
        # the batch size when performing audio mixing
        B, C, T = wavs.shape
        k = int(mix_p * B)
        wav_idx = torch.randperm(B)[:k]
        wavs = wavs[wav_idx]
        infos = [infos[i] for i in wav_idx]
        assert wavs.shape[0] == len(infos), "Mismatch between number of wavs and infos in the batch"

        return wavs, infos  # [B, C, T]
