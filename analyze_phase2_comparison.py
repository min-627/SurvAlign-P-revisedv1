# -*- coding: utf-8 -*-
"""Paired statistical comparison for phase2_training.py evaluation outputs.

Consumes the per-sample CSVs (`*_samples.csv`) that `phase2_training.py`'s
`evaluate()` writes into a results directory, one file per (seed, mode). For a
single results directory it reports baseline-vs-method paired statistics; given
a second directory it additionally reports method-vs-method paired statistics
between the two runs, plus the Jaccard similarity of the two runs' "recovered"
sample sets.

All comparisons are paired by matching (seed, sample_id) exactly -- never by
row position or aggregate means -- so a seed whose test split differs in size
or membership from another seed's never gets silently misaligned.

Usage:
    python analyze_phase2_comparison.py --dir1 results/A --attack facodec
    python analyze_phase2_comparison.py --dir1 results/A --dir2 results/B --attack facodec
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

SAMPLES_SUFFIX = "_samples.csv"
SUMMARY_SUFFIX = "_summary.json"


def load_results_dir(dir_path: str) -> pd.DataFrame:
    """Load every `*_samples.csv` in dir_path and tag each row with its run's seed.

    The seed is read from the sibling `*_summary.json`'s `config.seed`, not parsed
    out of the filename: the stem is `{run_id}_{dataset_type}_{mode}_{map_type}`,
    and dataset_type/mode/map_type all routinely contain underscores themselves
    (e.g. "proposed_gate", "codec_utility"), so splitting the filename on "_" is
    ambiguous. The summary json is the one place that field is unambiguous.
    """
    pattern = os.path.join(dir_path, f"*{SAMPLES_SUFFIX}")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {SAMPLES_SUFFIX} files found in {dir_path!r}")

    frames = []
    for samples_path in paths:
        stem = samples_path[: -len(SAMPLES_SUFFIX)]
        summary_path = stem + SUMMARY_SUFFIX
        if not os.path.exists(summary_path):
            raise FileNotFoundError(
                f"Missing sibling summary file for {samples_path!r}: expected {summary_path!r}"
            )
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        config = summary.get("config", {})
        seed = config.get("seed")
        if seed is None:
            raise ValueError(f"{summary_path!r} has no config.seed; cannot pool runs by seed")
        df = pd.read_csv(samples_path)
        df["seed"] = seed
        df["mode"] = config.get("mode")
        df["map_type"] = config.get("map_type")
        df["source_file"] = os.path.basename(samples_path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    n_seeds = combined["seed"].nunique()
    print(f"[load] {dir_path}: {len(paths)} run file(s), {n_seeds} distinct seed(s), "
          f"{len(combined)} sample-rows total")
    return combined


def filter_attack(df: pd.DataFrame, attack: str) -> pd.DataFrame:
    subset = df[df["attack"] == attack]
    if subset.empty:
        available = sorted(df["attack"].unique())
        raise ValueError(f"No rows for attack={attack!r} in this directory. Available: {available}")
    return subset


def pair_systems(df: pd.DataFrame, system_a: str, system_b: str) -> pd.DataFrame:
    """Inner-join system_a's and system_b's rows on (seed, sample_id).

    This is the paired-matching step: it guarantees every row in the result
    compares the *same underlying sample* (same seed's test split, same
    sample_id) between the two systems, and drops anything that doesn't have
    a counterpart on both sides (reporting how many rows that affected).
    """
    a = df[df["system"] == system_a]
    b = df[df["system"] == system_b]
    if a.empty:
        raise ValueError(f"No rows with system={system_a!r}")
    if b.empty:
        raise ValueError(f"No rows with system={system_b!r}")
    merged = a.merge(
        b, on=["seed", "sample_id"], suffixes=("_a", "_b"), how="inner", validate="one_to_one"
    )
    if merged.empty:
        raise ValueError(
            f"No overlapping (seed, sample_id) pairs between system={system_a!r} and system={system_b!r}"
        )
    dropped_a, dropped_b = len(a) - len(merged), len(b) - len(merged)
    if dropped_a or dropped_b:
        print(f"[WARNING] pairing system={system_a!r} vs system={system_b!r}: "
              f"dropped {dropped_a} unmatched row(s) from {system_a!r}, "
              f"{dropped_b} unmatched row(s) from {system_b!r}")
    return merged


def mcnemar_exact(exact_a: np.ndarray, exact_b: np.ndarray) -> Dict[str, float]:
    """Exact (binomial sign test) McNemar test on paired exact-message correctness."""
    exact_a = np.asarray(exact_a).astype(bool)
    exact_b = np.asarray(exact_b).astype(bool)
    both_correct = int(np.sum(exact_a & exact_b))
    a_only = int(np.sum(exact_a & ~exact_b))   # a correct, b wrong
    b_only = int(np.sum(~exact_a & exact_b))   # a wrong, b correct
    both_wrong = int(np.sum(~exact_a & ~exact_b))
    n_discordant = a_only + b_only
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            stats.binomtest(min(a_only, b_only), n_discordant, 0.5, alternative="two-sided").pvalue
        )
    return {
        "both_correct": both_correct,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_wrong": both_wrong,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }


def wilcoxon_bit_accuracy(bit_acc_a: np.ndarray, bit_acc_b: np.ndarray) -> Dict[str, float]:
    """Wilcoxon signed-rank test on paired per-sample bit_accuracy (b - a)."""
    bit_acc_a = np.asarray(bit_acc_a, dtype=np.float64)
    bit_acc_b = np.asarray(bit_acc_b, dtype=np.float64)
    diff = bit_acc_b - bit_acc_a
    n_nonzero = int(np.sum(diff != 0))
    mean_diff = float(np.mean(diff))
    if n_nonzero == 0:
        return {"statistic": float("nan"), "p_value": 1.0, "n_nonzero": 0, "mean_diff": mean_diff}
    statistic, p_value = stats.wilcoxon(bit_acc_a, bit_acc_b, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_nonzero": n_nonzero,
        "mean_diff": mean_diff,
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta between the distributions of x and y (positive => x tends larger).

    delta = (#{x_i > y_j} - #{x_i < y_j}) / (n * m), summed over all n*m pairs.
    Computed via the Mann-Whitney U statistic (O(n log n)) rather than the naive
    O(n*m) double loop: U1 (for x) already equals #{x_i > y_j} + 0.5*#{x_i == y_j}
    summed over all pairs, so delta = 2*U1/(n*m) - 1.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("nan")
    u_statistic, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float((2.0 * u_statistic) / (n * m) - 1.0)


def recovery_regression(exact_a: np.ndarray, exact_b: np.ndarray) -> Dict[str, float]:
    """Same definition as experiment_utils.recovery_regression_metrics, but on the
    paired 'exact' column already computed by phase2_training.py's evaluate().
    a is treated as the reference ("baseline"); recovery = a-failure fixed by b,
    regression = a-success broken by b.
    """
    exact_a = np.asarray(exact_a).astype(bool)
    exact_b = np.asarray(exact_b).astype(bool)
    a_fail = ~exact_a
    a_success = exact_a
    recovered = a_fail & exact_b
    regressed = a_success & (~exact_b)
    return {
        "recovery_rate": float(recovered.sum() / max(1, a_fail.sum())),
        "regression_rate": float(regressed.sum() / max(1, a_success.sum())),
        "n_a_failures": int(a_fail.sum()),
        "n_a_successes": int(a_success.sum()),
        "n_recovered": int(recovered.sum()),
        "n_regressed": int(regressed.sum()),
    }


def paired_comparison(paired: pd.DataFrame, delta_direction: str = "b_vs_a") -> Dict[str, object]:
    """delta_direction picks which way `cliffs_delta_bit_accuracy` (and mean_diff's
    sign) point: 'b_vs_a' (default) reads as "how much b improves over a" -- the
    natural framing for baseline(a) vs method(b). 'a_vs_b' reads as "a relative to
    b" -- used for method-vs-method so the delta's sign matches the order the two
    directories were named on the command line (--dir1 vs --dir2).
    """
    if delta_direction not in {"a_vs_b", "b_vs_a"}:
        raise ValueError(f"Unknown delta_direction: {delta_direction!r}")
    exact_a = paired["exact_a"].to_numpy()
    exact_b = paired["exact_b"].to_numpy()
    bit_a = paired["bit_accuracy_a"].to_numpy(dtype=np.float64)
    bit_b = paired["bit_accuracy_b"].to_numpy(dtype=np.float64)
    if delta_direction == "b_vs_a":
        delta = cliffs_delta(bit_b, bit_a)
        mean_diff = float(np.mean(bit_b - bit_a))
    else:
        delta = cliffs_delta(bit_a, bit_b)
        mean_diff = float(np.mean(bit_a - bit_b))
    return {
        "n_pairs": int(len(paired)),
        "mean_bit_accuracy_a": float(np.mean(bit_a)),
        "mean_bit_accuracy_b": float(np.mean(bit_b)),
        "mean_exact_accuracy_a": float(np.mean(exact_a)),
        "mean_exact_accuracy_b": float(np.mean(exact_b)),
        "mcnemar_exact": mcnemar_exact(exact_a, exact_b),
        "wilcoxon_bit_accuracy": wilcoxon_bit_accuracy(bit_a, bit_b),
        "cliffs_delta_bit_accuracy": delta,
        "cliffs_delta_direction": delta_direction,
        "mean_bit_accuracy_diff": mean_diff,
        "recovery_regression": recovery_regression(exact_a, exact_b),
    }


def recovered_sample_set(df: pd.DataFrame, baseline_system: str, method_system: str) -> Set[Tuple]:
    """(seed, sample_id) pairs where baseline's exact-message decode failed but
    method's succeeded, for whatever attack `df` has already been filtered to."""
    paired = pair_systems(df, baseline_system, method_system)
    mask = (~paired["exact_a"].astype(bool)) & paired["exact_b"].astype(bool)
    recovered = paired.loc[mask, ["seed", "sample_id"]]
    return set(zip(recovered["seed"], recovered["sample_id"]))


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    union = set_a | set_b
    if not union:
        return float("nan")
    return len(set_a & set_b) / len(union)


def print_stats(label: str, result: Dict[str, object], name_a: str, name_b: str) -> None:
    mc = result["mcnemar_exact"]
    wx = result["wilcoxon_bit_accuracy"]
    print(f"\n--- {label} ---")
    print(f"n_pairs={result['n_pairs']}")
    print(f"mean bit_accuracy: {name_a}={result['mean_bit_accuracy_a']:.6f}  "
          f"{name_b}={result['mean_bit_accuracy_b']:.6f}")
    print(f"mean exact_accuracy: {name_a}={result['mean_exact_accuracy_a']:.6f}  "
          f"{name_b}={result['mean_exact_accuracy_b']:.6f}")
    print(f"McNemar (exact, on exact-message correctness): "
          f"{name_a}-only={mc['a_only_correct']} {name_b}-only={mc['b_only_correct']} "
          f"discordant={mc['n_discordant']} p={mc['p_value']:.4f}")
    print(f"Wilcoxon (paired bit_accuracy, {name_b}-{name_a}): "
          f"statistic={wx['statistic']:.4f} p={wx['p_value']:.4f} "
          f"mean_diff={wx['mean_diff']:+.6f} n_nonzero={wx['n_nonzero']}")
    delta_a, delta_b = (name_a, name_b) if result["cliffs_delta_direction"] == "a_vs_b" else (name_b, name_a)
    print(f"Cliff's delta (bit_accuracy, {delta_a} vs {delta_b}): "
          f"{result['cliffs_delta_bit_accuracy']:+.4f} "
          f"(mean_bit_accuracy_diff={result['mean_bit_accuracy_diff']:+.6f})")
    rr = result["recovery_regression"]
    print(f"Recovery/regression ({name_a}->{name_b}): "
          f"recovery_rate={rr['recovery_rate']:.4f} ({rr['n_recovered']}/{rr['n_a_failures']})  "
          f"regression_rate={rr['regression_rate']:.4f} ({rr['n_regressed']}/{rr['n_a_successes']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired statistical comparison of phase2_training.py results")
    parser.add_argument("--dir1", required=True, help="Results directory (may contain several seeds' *_samples.csv)")
    parser.add_argument("--dir2", default="", help="Optional second results directory for method-vs-method comparison")
    parser.add_argument("--attack", required=True, help="Attack name to compare (must appear in the samples CSVs)")
    parser.add_argument("--baseline_system", default="baseline")
    parser.add_argument("--method_system", default="method")
    parser.add_argument("--output_json", default="", help="Optional path to dump the full report as JSON")
    args = parser.parse_args()

    report: Dict[str, object] = {"attack": args.attack, "dir1": args.dir1}

    df1 = load_results_dir(args.dir1)
    subset1 = filter_attack(df1, args.attack)
    pair1 = pair_systems(subset1, args.baseline_system, args.method_system)
    stats1 = paired_comparison(pair1)
    report["dir1_baseline_vs_method"] = stats1
    print_stats(f"{args.dir1}: {args.baseline_system} vs {args.method_system} ({args.attack})",
                stats1, args.baseline_system, args.method_system)

    if args.dir2:
        report["dir2"] = args.dir2
        df2 = load_results_dir(args.dir2)
        subset2 = filter_attack(df2, args.attack)
        pair2 = pair_systems(subset2, args.baseline_system, args.method_system)
        stats2 = paired_comparison(pair2)
        report["dir2_baseline_vs_method"] = stats2
        print_stats(f"{args.dir2}: {args.baseline_system} vs {args.method_system} ({args.attack})",
                    stats2, args.baseline_system, args.method_system)

        method1 = subset1[subset1["system"] == args.method_system]
        method2 = subset2[subset2["system"] == args.method_system]
        merged_methods = method1.merge(
            method2, on=["seed", "sample_id"], suffixes=("_a", "_b"), how="inner", validate="one_to_one"
        )
        if merged_methods.empty:
            raise ValueError(
                "No overlapping (seed, sample_id) pairs between dir1 and dir2 method rows -- "
                "the two runs must share seeds/test splits to be paired. "
                f"dir1 seeds={sorted(subset1['seed'].unique())}, dir2 seeds={sorted(subset2['seed'].unique())}"
            )
        dropped_a = len(method1) - len(merged_methods)
        dropped_b = len(method2) - len(merged_methods)
        if dropped_a or dropped_b:
            print(f"[WARNING] method-vs-method pairing dropped {dropped_a} row(s) from dir1, "
                  f"{dropped_b} row(s) from dir2 (seeds/sample_ids without a counterpart)")
        cross_stats = paired_comparison(merged_methods, delta_direction="a_vs_b")
        report["method_vs_method"] = cross_stats
        label_a, label_b = f"method[{args.dir1}]", f"method[{args.dir2}]"
        print_stats(f"method({args.dir1}) vs method({args.dir2}), attack={args.attack}",
                    cross_stats, label_a, label_b)

        recovered1 = recovered_sample_set(subset1, args.baseline_system, args.method_system)
        recovered2 = recovered_sample_set(subset2, args.baseline_system, args.method_system)
        jaccard = jaccard_similarity(recovered1, recovered2)
        report["recovered_set_jaccard"] = {
            "n_recovered_dir1": len(recovered1),
            "n_recovered_dir2": len(recovered2),
            "n_intersection": len(recovered1 & recovered2),
            "n_union": len(recovered1 | recovered2),
            "jaccard": jaccard,
        }
        print(f"\n--- Recovered-sample-set Jaccard similarity ({args.attack}) ---")
        print(f"|recovered({args.dir1})|={len(recovered1)}  |recovered({args.dir2})|={len(recovered2)}  "
              f"|intersection|={len(recovered1 & recovered2)}  |union|={len(recovered1 | recovered2)}  "
              f"jaccard={jaccard:.4f}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=True)
        print(f"\nSaved full report: {args.output_json}")


if __name__ == "__main__":
    main()
