#!/usr/bin/env python3
"""Paired comparison of FailureSensorIQ validation prediction files."""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=path/to/val_predictions.csv")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Run name cannot be empty")
    return name, Path(path).expanduser().resolve()


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    values = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        values[i] = b[idx].mean() - a[idx].mean()
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def load_run(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    required = {"id", "gold", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)} in {path}")
    df = df[["id", "gold", "prediction"]].copy()
    if df["id"].duplicated().any():
        raise ValueError(f"{name}: duplicate ids")
    df[f"correct::{name}"] = (df["gold"] == df["prediction"]).astype(int)
    return df.rename(columns={"prediction": f"prediction::{name}"})


def main() -> None:
    p = argparse.ArgumentParser(description="Compare base/QLoRA/CSReg predictions with paired tests")
    p.add_argument("--run", action="append", required=True, type=parse_run, help="NAME=val_predictions.csv")
    p.add_argument("--output-dir", default="outputs/ablation_comparison")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    runs = args.run
    merged: pd.DataFrame | None = None
    for name, path in runs:
        frame = load_run(name, path)
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame.drop(columns=["gold"]), on="id", how="inner", validate="one_to_one")
    assert merged is not None
    if len(merged) == 0:
        raise ValueError("No common validation ids across runs")

    summary = []
    for name, _ in runs:
        correct = merged[f"correct::{name}"].to_numpy(dtype=np.int64)
        summary.append({"run": name, "n": len(correct), "accuracy": float(correct.mean())})

    pairs = []
    for (name_a, _), (name_b, _) in combinations(runs, 2):
        a = merged[f"correct::{name_a}"].to_numpy(dtype=np.int64)
        b = merged[f"correct::{name_b}"].to_numpy(dtype=np.int64)
        b_only = int(((a == 0) & (b == 1)).sum())
        a_only = int(((a == 1) & (b == 0)).sum())
        low, high = paired_bootstrap_delta(a, b, seed=args.seed, n_boot=args.bootstrap)
        pairs.append(
            {
                "run_a": name_a,
                "run_b": name_b,
                "accuracy_a": float(a.mean()),
                "accuracy_b": float(b.mean()),
                "delta_b_minus_a": float(b.mean() - a.mean()),
                "paired_bootstrap_95_ci": [low, high],
                "a_wrong_b_correct": b_only,
                "a_correct_b_wrong": a_only,
                "mcnemar_exact_p": exact_mcnemar_p(b_only, a_only),
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"n_common": len(merged), "runs": summary, "pairwise": pairs}
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(out_dir / "run_summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(out_dir / "pairwise_comparison.csv", index=False)
    merged.to_csv(out_dir / "paired_predictions.csv", index=False)

    lines = ["# FailureSensorIQ ablation comparison", "", "## Accuracy", "", "| Run | N | Accuracy |", "|---|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['run']} | {row['n']} | {100*row['accuracy']:.2f}% |")
    lines += ["", "## Paired comparisons", "", "| A | B | B − A | 95% CI | McNemar p |", "|---|---|---:|---:|---:|"]
    for row in pairs:
        ci = row["paired_bootstrap_95_ci"]
        lines.append(
            f"| {row['run_a']} | {row['run_b']} | {100*row['delta_b_minus_a']:.2f}%p | "
            f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]%p | {row['mcnemar_exact_p']:.4g} |"
        )
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
