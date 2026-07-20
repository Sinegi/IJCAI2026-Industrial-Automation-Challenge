#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path.cwd()
for candidate in (PROJECT_ROOT, ROOT):
    if (candidate / "fsiq_csreg").exists():
        sys.path.insert(0, str(candidate))
        break

try:
    from fsiq_csreg.core import build_inference_prompt, load_normalized_records
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Run this script from the FailureSensorIQ project root, where fsiq_csreg/ exists."
    ) from exc


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 3000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_difference_ci(
    a: np.ndarray, b: np.ndarray, seed: int, n_boot: int = 3000
) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        diffs[i] = (
            rng.choice(a, size=len(a), replace=True).mean()
            - rng.choice(b, size=len(b), replace=True).mean()
        )
    point = float(a.mean() - b.mean())
    return point, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def safe_qcut(series: pd.Series, q: int, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        bins = pd.qcut(numeric, q=q, duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(series), index=series.index)
    categories = list(bins.cat.categories)
    mapping = {cat: f"{prefix}{i + 1}" for i, cat in enumerate(categories)}
    return bins.map(mapping).astype(str)


def group_metrics(df: pd.DataFrame, columns: list[str], seed: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouped = df.groupby(columns, dropna=False, observed=False)
    for key, part in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        ci_low, ci_high = bootstrap_mean_ci(part["correct"].to_numpy(), seed=seed)
        record = {name: value for name, value in zip(columns, key)}
        record.update(
            {
                "n": int(len(part)),
                "accuracy": float(part["correct"].mean()),
                "accuracy_percent": 100.0 * float(part["correct"].mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_prompt_tokens": float(part["prompt_tokens"].mean()),
                "mean_total_chars": float(part["total_chars"].mean()),
                "mean_n_options": float(part["n_options"].mean()),
            }
        )
        records.append(record)
    return pd.DataFrame(records).sort_values(["accuracy", "n"], ascending=[True, False])


def prediction_frame(path: str | Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str, "gold": str, "prediction": str})
    required = {"id", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.rename(columns={"prediction": f"prediction_{name}"})
    if "correct" in df.columns:
        df = df.rename(columns={"correct": f"correct_{name}"})
    keep = ["id", f"prediction_{name}"]
    if f"correct_{name}" in df.columns:
        keep.append(f"correct_{name}")
    return df[keep]


def paired_comparison(df: pd.DataFrame, model_a: str, model_b: str, seed: int) -> dict[str, Any]:
    a = df[f"correct_{model_a}"].astype(int).to_numpy()
    b = df[f"correct_{model_b}"].astype(int).to_numpy()
    diff = b - a
    rng = np.random.default_rng(seed)
    boot = np.empty(5000, dtype=np.float64)
    indices = np.arange(len(diff))
    for i in range(len(boot)):
        sampled = rng.choice(indices, size=len(indices), replace=True)
        boot[i] = diff[sampled].mean()
    return {
        "model_a": model_a,
        "model_b": model_b,
        "n": int(len(df)),
        "accuracy_a": float(a.mean()),
        "accuracy_b": float(b.mean()),
        "delta_b_minus_a": float(diff.mean()),
        "paired_bootstrap_95_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "both_correct": int(np.sum((a == 1) & (b == 1))),
        "a_only_correct": int(np.sum((a == 1) & (b == 0))),
        "b_only_correct": int(np.sum((a == 0) & (b == 1))),
        "both_wrong": int(np.sum((a == 0) & (b == 0))),
    }


def load_tokenizer(model_id: str | None):
    if not model_id:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception as exc:
        print(f"[warning] Could not load tokenizer {model_id!r}: {exc}")
        print("[warning] Token-length diagnostics will fall back to whitespace word counts.")
        return None


def label_position(label: str, options: OrderedDict[str, str]) -> int | None:
    labels = list(options)
    try:
        return labels.index(str(label).upper()) + 1
    except ValueError:
        return None


def make_charts(df: pd.DataFrame, tables: dict[str, pd.DataFrame], output_dir: Path, model_name: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warning] matplotlib unavailable; skipping charts: {exc}")
        return

    def save_bar(table: pd.DataFrame, x: str, title: str, filename: str, rotate: int = 0) -> None:
        if table.empty:
            return
        ordered = table.copy()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(ordered[x].astype(str), ordered["accuracy_percent"])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=rotate)
        for i, row in ordered.reset_index(drop=True).iterrows():
            ax.text(i, row["accuracy_percent"] + 1.0, f"{row['accuracy_percent']:.1f}\n(n={int(row['n'])})", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    save_bar(tables["by_polarity"].sort_values("polarity"), "polarity", f"{model_name}: accuracy by polarity", "accuracy_by_polarity.png")
    save_bar(tables["by_direction"].sort_values("direction"), "direction", f"{model_name}: accuracy by direction", "accuracy_by_direction.png")
    save_bar(tables["by_n_options"].sort_values("n_options"), "n_options", f"{model_name}: accuracy by number of choices", "accuracy_by_n_options.png")
    save_bar(tables["by_prompt_length_bin"].sort_values("prompt_length_bin"), "prompt_length_bin", f"{model_name}: accuracy by prompt-token quintile", "accuracy_by_prompt_length.png")
    save_bar(tables["by_gold_position"].sort_values("gold_position"), "gold_position", f"{model_name}: accuracy by gold answer position", "accuracy_by_gold_position.png")

    pos = tables["position_distribution"].sort_values("relative_position_bucket")
    if not pos.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(pos))
        width = 0.38
        ax.bar(x - width / 2, pos["gold_share_percent"], width, label="Gold")
        ax.bar(x + width / 2, pos["predicted_share_percent"], width, label="Predicted")
        ax.set_xticks(x)
        ax.set_xticklabels(pos["relative_position_bucket"])
        ax.set_ylabel("Share (%)")
        ax.set_title(f"{model_name}: gold vs predicted relative answer position")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "position_distribution.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CSReg validation errors by length, polarity, option count, and answer position.")
    parser.add_argument("--val", default="val.jsonl")
    parser.add_argument("--predictions", required=True, help="CSReg val_predictions.csv")
    parser.add_argument("--base-predictions", default=None, help="Optional base val_predictions.csv")
    parser.add_argument("--model-name", default="csreg")
    parser.add_argument("--base-name", default="base")
    parser.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--output-dir", default="outputs/csreg_error_analysis")
    parser.add_argument("--length-bins", type=int, default=5)
    parser.add_argument("--training-max-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-errors", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [row for row in load_normalized_records(args.val) if row.is_single_answer]
    row_by_id = {str(row.id): row for row in rows}
    pred = prediction_frame(args.predictions, args.model_name)
    if pred["id"].duplicated().any():
        raise ValueError("Duplicate ids in predictions")

    tokenizer = load_tokenizer(args.tokenizer)
    feature_records: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    for rec in pred.to_dict("records"):
        row = row_by_id.get(str(rec["id"]))
        if row is None:
            missing_ids.append(str(rec["id"]))
            continue
        gold = str(row.answer_label).upper()
        prediction = str(rec[f"prediction_{args.model_name}"]).upper()
        options = row.options
        labels = list(options)
        gold_pos = label_position(gold, options)
        pred_pos = label_position(prediction, options)
        n_options = len(options)
        gold_norm = (gold_pos - 1) / max(n_options - 1, 1) if gold_pos else np.nan
        pred_norm = (pred_pos - 1) / max(n_options - 1, 1) if pred_pos else np.nan
        correct_option = options.get(gold, "")
        distractors = [text for label, text in options.items() if label != gold]
        passage = row.passage or ""
        question = row.question or ""
        option_block = "\n".join(f"{label}. {text}" for label, text in options.items())
        prompt_text = f"{passage}\n\n{question}\n\n{option_block}".strip()
        if tokenizer is not None:
            exact_inference_prompt = build_inference_prompt(row, tokenizer)
            prompt_tokens = len(tokenizer(exact_inference_prompt, add_special_tokens=False)["input_ids"])
        else:
            prompt_tokens = len(prompt_text.split())
        feature_records.append(
            {
                "id": str(row.id),
                "gold": gold,
                "prediction": prediction,
                "correct": int(prediction == gold),
                "asset": row.asset,
                "family": row.metadata.get("family", row.relevancy or row.question_type),
                "question_type": row.question_type,
                "direction": row.direction,
                "polarity": row.polarity,
                "anchor": row.anchor,
                "n_options": n_options,
                "gold_position": gold_pos,
                "predicted_position": pred_pos,
                "gold_position_normalized": gold_norm,
                "predicted_position_normalized": pred_norm,
                "passage_chars": len(passage),
                "question_chars": len(question),
                "options_chars": len(option_block),
                "total_chars": len(prompt_text),
                "passage_words": len(passage.split()),
                "question_words": len(question.split()),
                "options_words": len(option_block.split()),
                "prompt_words": len(prompt_text.split()),
                "prompt_tokens": prompt_tokens,
                "correct_option_chars": len(correct_option),
                "correct_option_words": len(correct_option.split()),
                "mean_distractor_chars": float(np.mean([len(x) for x in distractors])) if distractors else 0.0,
                "correct_minus_distractor_chars": len(correct_option) - (float(np.mean([len(x) for x in distractors])) if distractors else 0.0),
                "passage": passage,
                "question": question,
                "options_json": json.dumps(options, ensure_ascii=False),
            }
        )

    if missing_ids:
        raise ValueError(f"Prediction ids not found in val data: {missing_ids[:10]} (total={len(missing_ids)})")

    df = pd.DataFrame(feature_records)
    df["over_training_max_length"] = df["prompt_tokens"] > int(args.training_max_length)
    df["prompt_length_bin"] = safe_qcut(df["prompt_tokens"], args.length_bins, "Q")
    df["passage_length_bin"] = safe_qcut(df["passage_chars"], args.length_bins, "Q")
    df["question_length_bin"] = safe_qcut(df["question_chars"], args.length_bins, "Q")
    df["correct_option_length_bin"] = safe_qcut(df["correct_option_chars"], args.length_bins, "Q")
    df["relative_position_bucket"] = pd.cut(
        df["gold_position_normalized"],
        bins=[-0.01, 0.20, 0.40, 0.60, 0.80, 1.01],
        labels=["first 20%", "20–40%", "40–60%", "60–80%", "last 20%"],
        include_lowest=True,
    ).astype(str)
    df["predicted_relative_position_bucket"] = pd.cut(
        df["predicted_position_normalized"],
        bins=[-0.01, 0.20, 0.40, 0.60, 0.80, 1.01],
        labels=["first 20%", "20–40%", "40–60%", "60–80%", "last 20%"],
        include_lowest=True,
    ).astype(str)

    if args.base_predictions:
        base = prediction_frame(args.base_predictions, args.base_name)
        df = df.merge(base, on="id", how="left", validate="one_to_one")
        df[f"correct_{args.base_name}"] = (df[f"prediction_{args.base_name}"].str.upper() == df["gold"]).astype(int)
        df[f"correct_{args.model_name}"] = df["correct"]
        df["transition"] = np.select(
            [
                (df[f"correct_{args.base_name}"] == 1) & (df["correct"] == 1),
                (df[f"correct_{args.base_name}"] == 1) & (df["correct"] == 0),
                (df[f"correct_{args.base_name}"] == 0) & (df["correct"] == 1),
            ],
            ["both_correct", "base_only_correct", "csreg_only_correct"],
            default="both_wrong",
        )

    tables: dict[str, pd.DataFrame] = {}
    group_specs = {
        "by_polarity": ["polarity"],
        "by_direction": ["direction"],
        "by_family": ["family"],
        "by_asset": ["asset"],
        "by_n_options": ["n_options"],
        "by_gold_position": ["gold_position"],
        "by_relative_position": ["relative_position_bucket"],
        "by_prompt_length_bin": ["prompt_length_bin"],
        "by_over_training_max_length": ["over_training_max_length"],
        "by_passage_length_bin": ["passage_length_bin"],
        "by_question_length_bin": ["question_length_bin"],
        "by_correct_option_length_bin": ["correct_option_length_bin"],
        "by_polarity_direction": ["polarity", "direction"],
        "by_polarity_n_options": ["polarity", "n_options"],
        "by_direction_n_options": ["direction", "n_options"],
    }
    for name, columns in group_specs.items():
        tables[name] = group_metrics(df, columns, seed=args.seed)
        tables[name].to_csv(output_dir / f"{name}.csv", index=False)

    buckets = ["first 20%", "20–40%", "40–60%", "60–80%", "last 20%"]
    gold_dist = df["relative_position_bucket"].value_counts(normalize=True)
    pred_dist = df["predicted_relative_position_bucket"].value_counts(normalize=True)
    position_distribution = pd.DataFrame(
        {
            "relative_position_bucket": buckets,
            "gold_share": [float(gold_dist.get(x, 0.0)) for x in buckets],
            "predicted_share": [float(pred_dist.get(x, 0.0)) for x in buckets],
        }
    )
    position_distribution["gold_share_percent"] = 100.0 * position_distribution["gold_share"]
    position_distribution["predicted_share_percent"] = 100.0 * position_distribution["predicted_share"]
    position_distribution["prediction_minus_gold_pp"] = (
        position_distribution["predicted_share_percent"] - position_distribution["gold_share_percent"]
    )
    tables["position_distribution"] = position_distribution
    position_distribution.to_csv(output_dir / "position_distribution.csv", index=False)

    overall_ci = bootstrap_mean_ci(df["correct"].to_numpy(), seed=args.seed)
    neg = df.loc[df["polarity"] == "negative", "correct"].to_numpy()
    pos = df.loc[df["polarity"] == "positive", "correct"].to_numpy()
    neg_gap = bootstrap_difference_ci(neg, pos, seed=args.seed)
    q1 = df.loc[df["prompt_length_bin"] == "Q1", "correct"].to_numpy()
    last_bin = sorted(df["prompt_length_bin"].unique())[-1]
    qlast = df.loc[df["prompt_length_bin"] == last_bin, "correct"].to_numpy()
    long_gap = bootstrap_difference_ci(qlast, q1, seed=args.seed)
    fewest = int(df["n_options"].min())
    most = int(df["n_options"].max())
    option_gap = bootstrap_difference_ci(
        df.loc[df["n_options"] == most, "correct"].to_numpy(),
        df.loc[df["n_options"] == fewest, "correct"].to_numpy(),
        seed=args.seed,
    )
    tv_distance = 0.5 * float(np.abs(position_distribution["predicted_share"] - position_distribution["gold_share"]).sum())

    summary: dict[str, Any] = {
        "model": args.model_name,
        "n": int(len(df)),
        "accuracy": float(df["correct"].mean()),
        "accuracy_percent": 100.0 * float(df["correct"].mean()),
        "bootstrap_95_ci": list(overall_ci),
        "negative_minus_positive_accuracy": {
            "difference": neg_gap[0],
            "bootstrap_95_ci": [neg_gap[1], neg_gap[2]],
            "negative_n": int(len(neg)),
            "positive_n": int(len(pos)),
        },
        "longest_minus_shortest_prompt_quintile": {
            "difference": long_gap[0],
            "bootstrap_95_ci": [long_gap[1], long_gap[2]],
            "shortest_bin": "Q1",
            "longest_bin": last_bin,
        },
        "most_minus_fewest_choices": {
            "difference": option_gap[0],
            "bootstrap_95_ci": [option_gap[1], option_gap[2]],
            "fewest_choices": fewest,
            "most_choices": most,
        },
        "training_length_limit": {
            "configured_max_length": int(args.training_max_length),
            "rows_over_limit": int(df["over_training_max_length"].sum()),
            "share_over_limit": float(df["over_training_max_length"].mean()),
            "accuracy_over_limit": float(df.loc[df["over_training_max_length"], "correct"].mean()) if df["over_training_max_length"].any() else None,
            "accuracy_within_limit": float(df.loc[~df["over_training_max_length"], "correct"].mean()) if (~df["over_training_max_length"]).any() else None,
        },
        "position_bias": {
            "total_variation_distance_gold_vs_prediction": tv_distance,
            "mean_gold_position_normalized": float(df["gold_position_normalized"].mean()),
            "mean_predicted_position_normalized": float(df["predicted_position_normalized"].mean()),
            "mean_prediction_minus_gold_position": float((df["predicted_position_normalized"] - df["gold_position_normalized"]).mean()),
        },
        "notes": [
            "Static position diagnostics can reveal suspicious answer-position patterns, but causal position bias requires the separate permutation-stability test.",
            "Length bins are dataset-specific quantiles, so each bin has approximately equal sample count.",
        ],
    }

    if args.base_predictions:
        summary["paired_vs_base"] = paired_comparison(df, args.base_name, args.model_name, args.seed)
        for dimension in ["polarity", "direction", "family", "asset", "n_options", "prompt_length_bin", "gold_position"]:
            records = []
            for key, part in df.groupby(dimension, dropna=False):
                records.append(
                    {
                        dimension: key,
                        "n": int(len(part)),
                        "base_accuracy": float(part[f"correct_{args.base_name}"].mean()),
                        "csreg_accuracy": float(part["correct"].mean()),
                        "delta_csreg_minus_base": float(part["correct"].mean() - part[f"correct_{args.base_name}"].mean()),
                    }
                )
            pd.DataFrame(records).sort_values("delta_csreg_minus_base").to_csv(
                output_dir / f"comparison_by_{dimension}.csv", index=False
            )
        df["transition"].value_counts().rename_axis("transition").reset_index(name="n").to_csv(
            output_dir / "base_to_csreg_transitions.csv", index=False
        )

    df.to_csv(output_dir / "item_level_diagnostics.csv", index=False)
    wrong = df[df["correct"] == 0].copy()
    wrong.sort_values(["prompt_tokens", "n_options"], ascending=[False, False]).head(args.top_errors).to_csv(
        output_dir / "longest_incorrect_examples.csv", index=False
    )
    wrong[wrong["polarity"] == "negative"].sort_values("prompt_tokens", ascending=False).head(args.top_errors).to_csv(
        output_dir / "negative_incorrect_examples.csv", index=False
    )
    wrong.sort_values(["n_options", "prompt_tokens"], ascending=[False, False]).head(args.top_errors).to_csv(
        output_dir / "many_choice_incorrect_examples.csv", index=False
    )

    make_charts(df, tables, output_dir, args.model_name)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        f"# {args.model_name} validation error diagnostics",
        "",
        f"- N: {len(df)}",
        f"- Accuracy: {100.0 * df['correct'].mean():.2f}%",
        f"- 95% bootstrap CI: {100.0 * overall_ci[0]:.2f}% – {100.0 * overall_ci[1]:.2f}%",
        "",
        "## Key gaps",
        "",
        f"- Negative − positive accuracy: {100.0 * neg_gap[0]:+.2f} percentage points "
        f"(95% CI {100.0 * neg_gap[1]:+.2f} to {100.0 * neg_gap[2]:+.2f})",
        f"- Longest − shortest prompt-token quintile: {100.0 * long_gap[0]:+.2f} percentage points "
        f"(95% CI {100.0 * long_gap[1]:+.2f} to {100.0 * long_gap[2]:+.2f})",
        f"- {most} choices − {fewest} choices: {100.0 * option_gap[0]:+.2f} percentage points "
        f"(95% CI {100.0 * option_gap[1]:+.2f} to {100.0 * option_gap[2]:+.2f})",
        f"- Gold/predicted relative-position total variation distance: {tv_distance:.3f}",
        f"- Rows above training max length ({args.training_max_length} tokens): {int(df['over_training_max_length'].sum())} ({100.0 * df['over_training_max_length'].mean():.2f}%)",
        "",
        "## Interpretation",
        "",
        "- A confidence interval that excludes 0 suggests a systematic gap rather than ordinary sampling noise.",
        "- Static answer-position results are diagnostic only. Run evaluate_permutation_stability.py to test whether changing option order changes the selected option content.",
        "- Inspect the generated CSV files for family-, asset-, and interaction-level failures before changing the model.",
    ]
    if args.base_predictions:
        paired = summary["paired_vs_base"]
        report_lines.extend(
            [
                "",
                "## Paired comparison with base",
                "",
                f"- Base accuracy: {100.0 * paired['accuracy_a']:.2f}%",
                f"- CSReg accuracy: {100.0 * paired['accuracy_b']:.2f}%",
                f"- Paired delta: {100.0 * paired['delta_b_minus_a']:+.2f} percentage points",
                f"- Paired bootstrap 95% CI: {100.0 * paired['paired_bootstrap_95_ci'][0]:+.2f} to "
                f"{100.0 * paired['paired_bootstrap_95_ci'][1]:+.2f} percentage points",
                f"- CSReg-only correct: {paired['b_only_correct']}",
                f"- Base-only correct: {paired['a_only_correct']}",
            ]
        )
    (output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved diagnostics to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
