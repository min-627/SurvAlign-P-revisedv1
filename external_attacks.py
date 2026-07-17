# -*- coding: utf-8 -*-
"""Optional held-out attack adapters (ffmpeg, ClearerVoice, FACodec, etc.).

오사용 방지 주석 (2026-07-16, encodec/vocos 서브프로세스 방식이 24분39초까지
늘어졌던 사고 이후): 그 근본 원인은 "서브프로세스를 쓴다"가 아니라 "신경망
체크포인트를 호출마다 디스크에서 다시 로드한다"였다 (inprocess_attacks.py의
_MODEL_CACHE 패턴으로 해결됨). 이 파일의 command_roundtrip_batch 기반 구현은
모델 로딩이 필요 없는 가벼운 네이티브 도구(ffmpeg 등)에만 허용된다.
신경망 체크포인트가 필요한 공격을 새로 추가할 때는 반드시
inprocess_attacks.py의 `_MODEL_CACHE`/`_get_model()`/`prewarm()` 패턴을 따를 것.
새 공격 추가 시 n=20 정도의 소규모 배치로 먼저 속도를 측정하고, 샘플당 1초를
넘으면 병목 원인(특히 반복 모델 로딩 여부)을 먼저 확인할 것.
"""

from __future__ import annotations

import multiprocessing
import os
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from experiment_utils import align_audio_tensors


def _soundfile():
    """Lazily import soundfile so that merely importing this module (e.g. for the
    asset-free smoke tests) does not require the native libsndfile dependency."""
    import soundfile as sf
    return sf


def _run_command_template(command_template: str, input_path: str, output_path: str) -> None:
    if "{input}" not in command_template or "{output}" not in command_template:
        raise ValueError("External command must contain both {input} and {output} placeholders.")
    command = command_template.format(input=input_path, output=output_path)
    result = subprocess.run(
        shlex.split(command, posix=(os.name != "nt")),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"External attack command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"External attack did not create output: {output_path}")


def command_roundtrip_batch(
    wav: torch.Tensor,
    command_template: str,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Apply an arbitrary file-based command independently to each waveform."""
    was_3d = wav.dim() == 3
    wav_2d = wav.squeeze(1) if was_3d else wav
    if wav_2d.dim() != 2:
        raise ValueError(f"Expected (B,T) or (B,1,T), got {wav.shape}")
    outputs = []
    with tempfile.TemporaryDirectory(prefix="survalign_attack_") as temp_dir:
        for index, sample in enumerate(wav_2d):
            input_path = os.path.join(temp_dir, f"input_{index}.wav")
            output_path = os.path.join(temp_dir, f"output_{index}.wav")
            sf = _soundfile()
            sf.write(input_path, sample.detach().cpu().numpy(), sample_rate, subtype="PCM_16")
            _run_command_template(command_template, input_path, output_path)
            output, sr = sf.read(output_path, dtype="float32")
            if output.ndim > 1:
                output = output.mean(axis=1)
            output_t = torch.from_numpy(np.asarray(output)).to(sample.device, sample.dtype)
            if sr != sample_rate:
                import torchaudio.functional as AF
                output_t = AF.resample(output_t.unsqueeze(0), sr, sample_rate).squeeze(0)
            sample_aligned, output_aligned = align_audio_tensors(sample, output_t)
            if output_aligned.shape[-1] < sample.shape[-1]:
                output_aligned = torch.nn.functional.pad(output_aligned, (0, sample.shape[-1] - output_aligned.shape[-1]))
            outputs.append(output_aligned[..., : sample.shape[-1]])
    stacked = torch.stack(outputs, dim=0)
    return stacked.unsqueeze(1) if was_3d else stacked


def _ffmpeg_codec_roundtrip_batch(
    wav: torch.Tensor,
    codec_name: str,
    codec_args: list,
    file_ext: str,
    sample_rate: int = 16000,
    bitrate: str = "64k",
    max_workers: int = None,
) -> torch.Tensor:
    """Apply a real ffmpeg encode/decode round trip, parallelized across samples.

    Each sample's ffmpeg subprocess is fully independent I/O-bound work (no shared
    model state, unlike encodec/vocos/facodec), so thread-level parallelism is safe
    and effective: since ffmpeg subprocesses don't hold the GIL, wall-clock speedup
    scales roughly with CPU core count. Shared by ffmpeg_mp3_roundtrip_batch and
    ffmpeg_aac_roundtrip_batch. Batch order is preserved explicitly via the `index`
    carried through `_process_one`, independent of any ordering guarantee from the
    executor itself.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError(f"ffmpeg was not found on PATH; cannot run real {codec_name} evaluation.")
    was_3d = wav.dim() == 3
    wav_2d = wav.squeeze(1) if was_3d else wav
    workers = max_workers or min(8, multiprocessing.cpu_count())

    def _process_one(item):
        index, sample = item
        with tempfile.TemporaryDirectory(prefix=f"survalign_{codec_name}_{index}_") as temp_dir:
            input_path = os.path.join(temp_dir, "input.wav")
            compressed_path = os.path.join(temp_dir, f"compressed.{file_ext}")
            output_path = os.path.join(temp_dir, "output.wav")
            sf = _soundfile()
            sf.write(input_path, sample.detach().cpu().numpy(), sample_rate, subtype="PCM_16")
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", input_path,
                 *codec_args, "-b:a", bitrate, compressed_path],
                check=True,
            )
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", compressed_path,
                 "-ar", str(sample_rate), "-ac", "1", output_path],
                check=True,
            )
            output, _ = sf.read(output_path, dtype="float32")
            if output.ndim > 1:
                output = output.mean(axis=1)
            output_t = torch.from_numpy(np.asarray(output)).to(sample.device, sample.dtype)
            _, output_t = align_audio_tensors(sample, output_t)
            if output_t.shape[-1] < sample.shape[-1]:
                output_t = torch.nn.functional.pad(output_t, (0, sample.shape[-1] - output_t.shape[-1]))
            return index, output_t[..., : sample.shape[-1]]

    results = [None] * wav_2d.shape[0]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, output_t in executor.map(_process_one, enumerate(wav_2d)):
            results[index] = output_t

    stacked = torch.stack(results, dim=0)
    return stacked.unsqueeze(1) if was_3d else stacked


def ffmpeg_mp3_roundtrip_batch(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    bitrate: str = "64k",
    max_workers: int = None,
) -> torch.Tensor:
    """Apply a real ffmpeg MP3 encode/decode round trip, parallelized across samples."""
    return _ffmpeg_codec_roundtrip_batch(
        wav, codec_name="mp3", codec_args=["-codec:a", "libmp3lame"], file_ext="mp3",
        sample_rate=sample_rate, bitrate=bitrate, max_workers=max_workers,
    )


def ffmpeg_aac_roundtrip_batch(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    bitrate: str = "64k",
    max_workers: int = None,
) -> torch.Tensor:
    """Apply a real ffmpeg AAC encode/decode round trip, parallelized across samples."""
    return _ffmpeg_codec_roundtrip_batch(
        wav, codec_name="aac", codec_args=["-codec:a", "aac"], file_ext="aac",
        sample_rate=sample_rate, bitrate=bitrate, max_workers=max_workers,
    )
