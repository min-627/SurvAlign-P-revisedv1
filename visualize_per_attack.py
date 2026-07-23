# -*- coding: utf-8 -*-
"""공격별 스펙트로그램 변화 비교 시각화.

visualize_channels.py로 만든 Survival Map(sample=librispeech:test:30, speaker=2035)에서
시간축으로는 거의 변하지 않고 주파수 대역으로만 뚜렷하게 갈리는 띠 모양이 보였다.
lowpass/bandpass/highpass 같은 필터형 공격이 이진적인(0 또는 1) 주파수 마스크를 만들어
Survival Map 전체를 지배하고 있는 것은 아닌지, 같은 오디오 하나에 11개 공격을 개별
적용해서 스펙트로그램이 실제로 어떻게 달라지는지 한 판에서 눈으로 비교한다.

공격 적용은 survalign_p._apply_survival_attack_pair() -> experiment_utils.apply_eval_attack()
디스패치를 그대로 재사용한다 (새 dispatch 테이블을 또 만들지 않음 -- replacement/masking/
frame_shuffle/highpass/ffmpeg_mp3/ffmpeg_aac/encodec/vocos가 예전에 바로 이런 중복 dispatch
불일치로 "Unsupported survival-map attack" 에러를 냈던 이력이 있다).
"""

from __future__ import annotations

import argparse
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from types import SimpleNamespace

from experiment_utils import align_audio_tensors, retained_energy_ratio, stable_int_hash
from survalign_p import (
    AlignMarkManager,
    DifferentiableDistortion,
    UnifiedSpeechDataset,
    _apply_survival_attack_pair,
    stft_audio,
)

# Neural/quantized codecs whose encode step can flip a discrete codebook entry from a tiny
# input perturbation, making (attacked_wm - attacked_clean) larger than the original
# residual itself (retained_energy_ratio > 1.0). survalign_p.get_survival_map() already
# anticipates this and clamps the equivalent per-pixel ratio to [0, 1]; see the note
# printed at the end of main() for detail. Not a bug in retained_energy_ratio().
NEURAL_CODEC_ATTACKS = frozenset({"encodec", "vocos", "facodec"})

ATTACKS = (
    "replacement", "masking", "frame_shuffle",
    "lowpass", "bandpass", "highpass",
    "ffmpeg_mp3", "ffmpeg_aac", "encodec", "vocos", "facodec",
)
# get_survival_map()과 동일한 STFT 파라미터 (survalign_p.py 참고).
N_FFT = 256
HOP_LENGTH = 64


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def to_db(magnitude: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return 20.0 * np.log10(magnitude + eps)


def main():
    parser = argparse.ArgumentParser(description="Per-attack spectrogram comparison")
    parser.add_argument("--dataset_type", default="librispeech", choices=["librispeech", "vctk", "ljspeech", "combined"])
    parser.add_argument("--dataset_name", default="dev-clean")
    parser.add_argument("--combined_protocol", default="speaker_disjoint", choices=["speaker_disjoint", "paper"])
    parser.add_argument("--latent_mode", default="public_code", choices=["public_code", "unquantized"])
    parser.add_argument("--split", default="test", choices=["train", "calib", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_index", type=int, default=30,
                        help="Dataset index; default matches visualize_channels.py's sample_0 (speaker=2035).")
    parser.add_argument("--attacks", default=",".join(ATTACKS))
    parser.add_argument("--mp3_bitrate", default="64k")
    parser.add_argument("--encodec_command", default="")
    parser.add_argument("--vocos_command", default="")
    parser.add_argument("--facodec_command", default="")
    parser.add_argument("--output_dir", default="outputs/per_attack_viz")
    args = parser.parse_args()

    attack_names = [a.strip() for a in args.attacks.split(",") if a.strip()]
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    needs_inprocess_model = (
        ("encodec" in attack_names and not args.encodec_command)
        or ("vocos" in attack_names and not args.vocos_command)
        or ("facodec" in attack_names and not args.facodec_command)
    )
    if needs_inprocess_model:
        from inprocess_attacks import prewarm

        prewarm(device)

    alignmark = AlignMarkManager(device, latent_mode=args.latent_mode)
    distorter = DifferentiableDistortion(sr=16000, vae=alignmark.vae).to(device)
    dataset = UnifiedSpeechDataset(
        dataset_type=args.dataset_type,
        dataset_name=args.dataset_name,
        split=args.split,
        seed=args.seed,
        return_metadata=True,
        combined_protocol=args.combined_protocol,
    )

    wav, msg, metadata = dataset[args.sample_index]
    wav = wav.unsqueeze(0).to(device)
    msg = msg.unsqueeze(0).to(device)

    wav_wm, residual = alignmark.embed(wav, msg)
    wav, wav_wm, residual = align_audio_tensors(wav, wav_wm, residual)

    eval_args = SimpleNamespace(
        mp3_bitrate=args.mp3_bitrate,
        encodec_command=args.encodec_command,
        vocos_command=args.vocos_command,
        facodec_command=args.facodec_command,
        clearervoice_command="",
        clearervoice_snr=10.0,
        dac_command="",
        hifigan_command="",
    )

    clean_db = to_db(np.abs(stft_audio(wav.squeeze(1), n_fft=N_FFT, hop_length=HOP_LENGTH)[0].cpu().numpy()))
    vmin, vmax = float(np.percentile(clean_db, 1)), float(np.percentile(clean_db, 99))

    panels = [("Original (pre-attack)", clean_db, None)]
    wm_db = to_db(np.abs(stft_audio(wav_wm.squeeze(1), n_fft=N_FFT, hop_length=HOP_LENGTH)[0].cpu().numpy()))
    panels.append(("Watermarked (no attack)", wm_db, None))

    energy_ratios = {}
    raw_energy_ratios = {}
    errors = {}
    for attack_name in attack_names:
        seed = stable_int_hash(args.seed, "per_attack", metadata["sample_id"], attack_name)
        try:
            attacked_clean, attacked_wm = _apply_survival_attack_pair(
                wav, wav_wm, distorter, attack_name, seed=seed, args=eval_args
            )
            attacked_clean, attacked_wm = align_audio_tensors(attacked_clean, attacked_wm)
            retained_residual, orig_residual = align_audio_tensors(
                attacked_wm - attacked_clean, residual
            )
            raw_ratio = float(retained_energy_ratio(retained_residual, orig_residual).item())
            # Match get_survival_map()'s own clamp(retention, 0.0, 1.0): for neural/quantized
            # codecs the raw ratio can exceed 1.0 (see NEURAL_CODEC_ATTACKS note above), which
            # would otherwise make the plotted "retained fraction" axis misleading.
            ratio = min(raw_ratio, 1.0)
            raw_energy_ratios[attack_name] = raw_ratio
            energy_ratios[attack_name] = ratio

            spec_db = to_db(np.abs(
                stft_audio(attacked_wm.squeeze(1), n_fft=N_FFT, hop_length=HOP_LENGTH)[0].detach().cpu().numpy()
            ))
            panels.append((attack_name, spec_db, None))
            if raw_ratio > 1.0:
                print(f"[OK] {attack_name}: retained_energy_ratio={ratio:.3f} (raw={raw_ratio:.3f}, capped at 1.0)")
            else:
                print(f"[OK] {attack_name}: retained_energy_ratio={ratio:.3f}")
        except Exception as exc:  # noqa: BLE001 -- per-attack failures must not abort the other 10
            errors[attack_name] = str(exc)
            panels.append((attack_name, None, str(exc)))
            print(f"[FAIL] {attack_name}: {exc}")
            traceback.print_exc()

    n_panels = len(panels)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.6 * n_rows))
    axes = np.atleast_2d(axes)
    for flat_index in range(n_rows * n_cols):
        ax = axes[flat_index // n_cols, flat_index % n_cols]
        if flat_index >= n_panels:
            ax.axis("off")
            continue
        title, spec_db, error = panels[flat_index]
        if error is not None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{title}\n[FAIL]\n{error[:80]}", ha="center", va="center", fontsize=8, wrap=True)
            continue
        im = ax.imshow(spec_db, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        subtitle = title
        if title in energy_ratios:
            subtitle += f"\nretained={energy_ratios[title]:.2f}"
        ax.set_title(subtitle, fontsize=9)
    fig.suptitle(f"sample={metadata['sample_id']}  speaker={metadata['speaker_id']}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96], h_pad=2.5)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="dB (common scale: 1st-99th percentile of the original spectrogram)")

    speaker_tag = sanitize(metadata["speaker_id"])
    grid_path = f"{args.output_dir}/sample_{args.sample_index}_{speaker_tag}_grid.png"
    fig.savefig(grid_path, dpi=160)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    filter_attacks = {"lowpass", "bandpass", "highpass"}
    ordered_attacks = [a for a in attack_names if a in energy_ratios]
    values = [energy_ratios[a] for a in ordered_attacks]
    colors = ["tab:red" if a in filter_attacks else "tab:blue" for a in ordered_attacks]
    hatches = ["//" if raw_energy_ratios[a] > 1.0 else None for a in ordered_attacks]
    bars = ax2.bar(ordered_attacks, values, color=colors)
    for bar, hatch, attack_name in zip(bars, hatches, ordered_attacks):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("black")
            ax2.text(bar.get_x() + bar.get_width() / 2, 1.02, f"raw={raw_energy_ratios[attack_name]:.1f}",
                      ha="center", va="bottom", fontsize=8)
    ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Retained watermark-residual energy ratio after attack (0-1, capped)")
    ax2.set_title(f"Retained residual energy per attack  sample={metadata['sample_id']} speaker={metadata['speaker_id']}\n"
                  f"(red = filter attacks lowpass/bandpass/highpass; hatched = raw ratio > 1.0, capped at 1.0)")
    ax2.tick_params(axis="x", rotation=45)
    for label in ax2.get_xticklabels():
        label.set_ha("right")
    fig2.tight_layout()
    bar_path = f"{args.output_dir}/sample_{args.sample_index}_{speaker_tag}_energy_bar.png"
    fig2.savefig(bar_path, dpi=160)
    plt.close(fig2)

    print(f"\n[SAVED] {grid_path}")
    print(f"[SAVED] {bar_path}")
    if errors:
        print(f"\n[SUMMARY] {len(errors)}/{len(attack_names)} attacks failed: {sorted(errors)}")
    else:
        print(f"\n[SUMMARY] all {len(attack_names)} attacks succeeded.")

    capped_attacks = {a: r for a, r in raw_energy_ratios.items() if r > 1.0}
    if capped_attacks:
        print("\n[NOTE] raw retained_energy_ratio > 1.0 for: "
              + ", ".join(f"{a}={r:.2f}" for a, r in capped_attacks.items()))
        print(
            "This is expected for learned/quantized codecs (encodec/vocos/facodec), not a bug in\n"
            "retained_energy_ratio(). Their encoder maps the waveform to *discrete* codebook\n"
            "indices; a tiny input perturbation (the watermark residual, usually much quieter\n"
            "than speech) can push a handful of frames across a codebook decision boundary and\n"
            "flip which entry gets selected. The decoder then reconstructs a completely different\n"
            "value at those frames -- a difference far larger than the input perturbation that\n"
            "caused it, because quantization is not smooth/Lipschitz-continuous at those\n"
            "boundaries. survalign_p.get_survival_map() already anticipates exactly this and\n"
            "clamps the equivalent per-pixel ratio to [0, 1] before using it as a survival score\n"
            "(see `retention = torch.clamp(retained_residual / residual_mag_safe, 0.0, 1.0)`).\n"
            "This script does the same (caps the plotted/reported ratio at 1.0) while still\n"
            "printing the raw, uncapped value above for transparency."
        )


if __name__ == "__main__":
    main()
