# -*- coding: utf-8 -*-
"""Experiment 3: Survival Map self-consistency via leave-one-attack-out.

For each attack A in the map-building set, we rebuild the Survival Map from the *remaining*
attacks and ask how well that map predicts where the residual actually survives attack A
(A was never used to build this map). Prediction quality is the per-sample Spearman rank
correlation between the leave-one-out map and the single-attack survival score of A.

Because A_surv typically contains only 5-6 attacks, the statistical power is low. Treat the
output as a qualitative stability check ("does the prior still point at surviving bins when a
family is withheld?") rather than a powered hypothesis test.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

from experiment_utils import save_json, set_global_seed, stable_int_hash
from survalign_p import get_survival_map


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _valid_support_mask(map_tensor: torch.Tensor, floor_quantile: float = 0.05) -> torch.Tensor:
    """Per-sample boolean mask discarding the lowest-magnitude bins (numerically unstable ranks)."""
    flat = map_tensor.reshape(map_tensor.shape[0], -1)
    floor = torch.quantile(flat, q=float(floor_quantile), dim=1).view(-1, 1, 1)
    return map_tensor > floor


def leave_one_attack_out_spearman(
    wav: torch.Tensor,
    wav_wm: torch.Tensor,
    distorter,
    survival_attacks: Sequence[str],
    base_seed: int = 42,
    quantile: float = 0.25,
) -> Dict[str, List[float]]:
    """Return {held_out_attack: [per-sample Spearman, ...]}.

    The leave-one-out map is built from ``survival_attacks \\ {held_out}`` and correlated
    against the single-attack survival score of ``held_out``.
    """
    from scipy.stats import spearmanr

    attacks = list(survival_attacks)
    if len(attacks) < 2:
        raise ValueError("Leave-one-attack-out requires at least 2 survival attacks.")

    results: Dict[str, List[float]] = {attack: [] for attack in attacks}
    for held_out in attacks:
        remaining = [a for a in attacks if a != held_out]
        loo_map = get_survival_map(
            wav, wav_wm, distorter, attack_names=remaining,
            base_seed=base_seed, quantile=quantile,
        )
        # Ground-truth "where does the residual actually survive held_out": the single-attack
        # survival score. Use a disjoint seed so it does not alias the leave-one-out seeds.
        target_map = get_survival_map(
            wav, wav_wm, distorter, attack_names=[held_out],
            base_seed=base_seed + 777_777, quantile=quantile,
        )
        support = _valid_support_mask(target_map)
        for item in range(loo_map.shape[0]):
            mask = support[item].reshape(-1).cpu().numpy().astype(bool)
            if mask.sum() < 3:
                continue
            x = loo_map[item].reshape(-1).cpu().numpy()[mask]
            y = target_map[item].reshape(-1).cpu().numpy()[mask]
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                continue
            rho = spearmanr(x, y).statistic
            if np.isfinite(rho):
                results[held_out].append(float(rho))
    return results


def summarize(results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for attack, values in results.items():
        if values:
            summary[attack] = {
                "spearman_mean": float(np.mean(values)),
                "spearman_std": float(np.std(values)),
                "spearman_median": float(np.median(values)),
                "n_valid": len(values),
            }
        else:
            summary[attack] = {
                "spearman_mean": float("nan"), "spearman_std": float("nan"),
                "spearman_median": float("nan"), "n_valid": 0,
            }
    all_values = [v for values in results.values() for v in values]
    summary["_overall"] = {
        "spearman_mean": float(np.mean(all_values)) if all_values else float("nan"),
        "spearman_std": float(np.std(all_values)) if all_values else float("nan"),
        "n_valid": len(all_values),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Survival Map leave-one-attack-out self-check")
    parser.add_argument("--dataset_type", default="librispeech", choices=["librispeech", "vctk", "ljspeech", "combined"])
    parser.add_argument("--dataset_name", default="dev-clean")
    parser.add_argument("--split", default="test", choices=["calib", "test"])
    parser.add_argument("--combined_protocol", default="speaker_disjoint", choices=["speaker_disjoint", "paper"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--survival_quantile", type=float, default=0.25)
    parser.add_argument("--survival_attacks", default="noise,lowpass,resample,speechtokenizer_nq6,spectral_proxy")
    parser.add_argument("--latent_mode", default="public_code", choices=["public_code", "unquantized"])
    parser.add_argument("--results_dir", default="results/phase1_experiment3")
    args = parser.parse_args()

    set_global_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Imported lazily so --help works without the AlignMark asset stack.
    from survalign_p import AlignMarkManager, DifferentiableDistortion, UnifiedSpeechDataset
    from torch.utils.data import DataLoader

    alignmark = AlignMarkManager(device, latent_mode=args.latent_mode)
    distorter = DifferentiableDistortion(sr=16000, vae=alignmark.vae).to(device)
    dataset = UnifiedSpeechDataset(
        dataset_type=args.dataset_type, dataset_name=args.dataset_name, split=args.split,
        seed=args.seed, return_metadata=True, combined_protocol=args.combined_protocol,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    survival_attacks = parse_csv_list(args.survival_attacks)

    aggregated: Dict[str, List[float]] = {attack: [] for attack in survival_attacks}
    processed = 0
    for batch_index, batch in enumerate(tqdm(loader, desc="Experiment3")):
        wav, msg, _ = batch
        if args.max_samples >= 0:
            remaining = args.max_samples - processed
            if remaining <= 0:
                break
            wav, msg = wav[:remaining], msg[:remaining]
        wav = wav.to(device)
        msg = msg.to(device)
        with torch.no_grad():
            wav_wm, _ = alignmark.embed(wav, msg)
        batch_results = leave_one_attack_out_spearman(
            wav, wav_wm, distorter, survival_attacks,
            base_seed=stable_int_hash(args.seed, batch_index),
            quantile=args.survival_quantile,
        )
        for attack, values in batch_results.items():
            aggregated[attack].extend(values)
        processed += wav.shape[0]

    summary = {"config": vars(args), "n_samples": processed, "leave_one_attack_out": summarize(aggregated)}
    save_json(os.path.join(args.results_dir, "experiment3_selfcheck.json"), summary)

    with open(os.path.join(args.results_dir, "experiment3_selfcheck.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["held_out_attack", "spearman_mean", "spearman_std", "spearman_median", "n_valid"])
        for attack in survival_attacks:
            stats = summary["leave_one_attack_out"][attack]
            writer.writerow([attack, stats["spearman_mean"], stats["spearman_std"],
                             stats["spearman_median"], stats["n_valid"]])

    print("\n[Experiment 3 completed] Leave-one-attack-out Survival-Map self-consistency")
    for attack in survival_attacks:
        stats = summary["leave_one_attack_out"][attack]
        print(f"  held-out {attack:<24} Spearman mean={stats['spearman_mean']:+.3f} "
              f"(median={stats['spearman_median']:+.3f}, n={stats['n_valid']})")
    overall = summary["leave_one_attack_out"]["_overall"]
    print(f"  overall Spearman mean={overall['spearman_mean']:+.3f} (n={overall['n_valid']})")
    print("  NOTE: A_surv is small (5-6 attacks); interpret as a qualitative stability check.")
    print(f"Results: {args.results_dir}")


if __name__ == "__main__":
    main()
