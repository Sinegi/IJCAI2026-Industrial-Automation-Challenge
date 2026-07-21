#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path.cwd()
if (PROJECT_ROOT / "fsiq_csreg").exists():
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fsiq_csreg.core import (
        CSRegPredictor,
        NormalizedMCQ,
        build_inference_prompt,
        canonical_entity,
        extract_answer_letter,
        load_normalized_records,
        stable_int,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Run this script from the FailureSensorIQ project root, where fsiq_csreg/ exists."
    ) from exc


def permute_row(row: NormalizedMCQ, permutation: list[int]) -> tuple[NormalizedMCQ, dict[str, str], dict[str, str]]:
    original_items = list(row.options.items())
    new_labels = list(row.options)
    permuted_contents = [original_items[index][1] for index in permutation]
    options = OrderedDict(zip(new_labels, permuted_contents))
    label_to_content = dict(options)
    content_to_original_label = {
        canonical_entity(content): original_label for original_label, content in original_items
    }
    gold_content = canonical_entity(row.options[row.answer_label])
    permuted_gold_label = next(
        label for label, content in options.items() if canonical_entity(content) == gold_content
    )
    permuted = NormalizedMCQ(
        id=row.id,
        question=row.question,
        options=options,
        passage=row.passage,
        asset=row.asset,
        relevancy=row.relevancy,
        question_type=row.question_type,
        direction=row.direction,
        polarity=row.polarity,
        anchor=row.anchor,
        correct_labels=[permuted_gold_label],
        reasoning=row.reasoning,
        metadata=dict(row.metadata),
    )
    return permuted, label_to_content, content_to_original_label


def deterministic_permutations(row: NormalizedMCQ, count: int, seed: int) -> list[list[int]]:
    n = len(row.options)
    first = list(range(n))
    perms = [first]
    seen = {tuple(first)}
    rng = random.Random(seed + stable_int(row.id))
    max_unique = math.factorial(n) if n <= 8 else count
    while len(perms) < count and len(seen) < max_unique:
        p = list(range(n))
        rng.shuffle(p)
        if tuple(p) not in seen:
            seen.add(tuple(p))
            perms.append(p)
    return perms


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 3000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    result = np.empty(n_boot)
    for i in range(n_boot):
        result[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(result, 0.025)), float(np.quantile(result, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate option-order sensitivity without aggregating TTA votes.")
    parser.add_argument("--config", default="configs/a100_40gb.yaml")
    parser.add_argument("--val", default="val.jsonl")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/csreg_permutation_stability")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rows = [row for row in load_normalized_records(args.val) if row.is_single_answer]
    if args.max_rows:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("No labeled single-answer rows found")

    predictor = CSRegPredictor(
        base_model_path=cfg["model"]["base_model"],
        adapter_path=args.adapter,
        load_in_4bit=True,
        max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
        tta_permutations=1,
        inference_batch_size=int(args.batch_size or cfg["inference"]["batch_size"]),
        seed=args.seed,
    )

    work: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for permutation_index, permutation in enumerate(
            deterministic_permutations(row, args.permutations, args.seed)
        ):
            permuted, label_to_content, content_to_original_label = permute_row(row, permutation)
            prompt = build_inference_prompt(permuted, predictor.tokenizer)
            gold_content = canonical_entity(row.options[row.answer_label])
            presented_gold_label = permuted.answer_label
            work.append(
                {
                    "row_index": row_index,
                    "row": row,
                    "permuted": permuted,
                    "permutation_index": permutation_index,
                    "permutation": permutation,
                    "label_to_content": label_to_content,
                    "content_to_original_label": content_to_original_label,
                    "prompt": prompt,
                    "gold_content": gold_content,
                    "presented_gold_label": presented_gold_label,
                }
            )

    records: list[dict[str, Any]] = []
    batch_size = predictor.inference_batch_size
    for start in range(0, len(work), batch_size):
        batch = work[start : start + batch_size]
        generated = predictor._generate_batch([item["prompt"] for item in batch])
        for item, text in zip(batch, generated):
            permuted = item["permuted"]
            answer = extract_answer_letter(text, list(permuted.options))
            used_fallback = False
            if answer is None:
                answer = predictor._score_label_candidates(permuted)
                used_fallback = True
            predicted_content = canonical_entity(item["label_to_content"][answer])
            original_prediction = item["content_to_original_label"].get(predicted_content)
            gold_label = item["row"].answer_label
            labels = list(permuted.options)
            presented_gold_position = labels.index(item["presented_gold_label"]) + 1
            presented_prediction_position = labels.index(answer) + 1
            records.append(
                {
                    "id": item["row"].id,
                    "permutation_index": item["permutation_index"],
                    "permutation_order": json.dumps(item["permutation"]),
                    "gold_original_label": gold_label,
                    "predicted_original_label": original_prediction,
                    "predicted_presented_label": answer,
                    "presented_gold_label": item["presented_gold_label"],
                    "presented_gold_position": presented_gold_position,
                    "presented_prediction_position": presented_prediction_position,
                    "n_options": len(permuted.options),
                    "correct": int(predicted_content == item["gold_content"]),
                    "asset": item["row"].asset,
                    "family": item["row"].metadata.get(
                        "family", item["row"].relevancy or item["row"].question_type
                    ),
                    "direction": item["row"].direction,
                    "polarity": item["row"].polarity,
                    "used_likelihood_fallback": int(used_fallback),
                    "generated_text": text,
                }
            )
        print(f"Processed {min(start + batch_size, len(work))}/{len(work)} permutation prompts")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "permutation_predictions.csv", index=False)

    per_item_records: list[dict[str, Any]] = []
    for item_id, part in df.groupby("id", sort=False):
        predictions = part["predicted_original_label"].fillna("INVALID").astype(str)
        counts = predictions.value_counts()
        majority_label = counts.index[0]
        majority_fraction = float(counts.iloc[0] / len(part))
        original = part.sort_values("permutation_index").iloc[0]
        per_item_records.append(
            {
                "id": item_id,
                "n_permutations": int(len(part)),
                "mean_accuracy_over_permutations": float(part["correct"].mean()),
                "original_order_correct": int(original["correct"]),
                "majority_prediction_original_label": majority_label,
                "majority_fraction": majority_fraction,
                "flip_rate": 1.0 - majority_fraction,
                "all_predictions_same": int(len(counts) == 1),
                "n_unique_predictions": int(len(counts)),
                "asset": original["asset"],
                "family": original["family"],
                "direction": original["direction"],
                "polarity": original["polarity"],
                "n_options": int(original["n_options"]),
            }
        )
    per_item = pd.DataFrame(per_item_records)
    per_item.to_csv(out_dir / "per_item_stability.csv", index=False)

    by_position = (
        df.groupby("presented_gold_position")["correct"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
    )
    by_position["accuracy_percent"] = 100.0 * by_position["accuracy"]
    by_position.to_csv(out_dir / "accuracy_by_presented_gold_position.csv", index=False)

    by_perm = (
        df.groupby("permutation_index")["correct"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
    )
    by_perm["accuracy_percent"] = 100.0 * by_perm["accuracy"]
    by_perm.to_csv(out_dir / "accuracy_by_permutation_index.csv", index=False)

    by_group_specs = {
        "stability_by_polarity.csv": "polarity",
        "stability_by_direction.csv": "direction",
        "stability_by_n_options.csv": "n_options",
        "stability_by_asset.csv": "asset",
        "stability_by_family.csv": "family",
    }
    for filename, column in by_group_specs.items():
        table = (
            per_item.groupby(column)
            .agg(
                n=("id", "count"),
                mean_flip_rate=("flip_rate", "mean"),
                fully_stable_share=("all_predictions_same", "mean"),
                mean_permutation_accuracy=("mean_accuracy_over_permutations", "mean"),
            )
            .reset_index()
            .sort_values("mean_flip_rate", ascending=False)
        )
        table.to_csv(out_dir / filename, index=False)

    original_acc = float(df.loc[df["permutation_index"] == 0, "correct"].mean())
    shuffled_acc = float(df.loc[df["permutation_index"] > 0, "correct"].mean())
    all_acc = float(df["correct"].mean())
    ci = bootstrap_mean_ci(per_item["mean_accuracy_over_permutations"].to_numpy(), args.seed)
    summary = {
        "n_items": int(len(per_item)),
        "permutations_per_item_requested": int(args.permutations),
        "total_prompt_evaluations": int(len(df)),
        "original_order_accuracy": original_acc,
        "shuffled_order_accuracy": shuffled_acc,
        "all_permutation_accuracy": all_acc,
        "per_item_mean_accuracy_bootstrap_95_ci": list(ci),
        "fully_stable_item_share": float(per_item["all_predictions_same"].mean()),
        "mean_flip_rate": float(per_item["flip_rate"].mean()),
        "median_flip_rate": float(per_item["flip_rate"].median()),
        "items_with_any_flip": int((per_item["flip_rate"] > 0).sum()),
        "likelihood_fallback_rate": float(df["used_likelihood_fallback"].mean()),
        "interpretation": {
            "fully_stable_item_share": "Fraction of questions for which the same option content was selected under every tested ordering.",
            "flip_rate": "For each question, 1 minus the vote share of the most common option content.",
            "shuffled_order_accuracy": "Accuracy after changing answer positions, before any TTA majority vote.",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(by_position["presented_gold_position"].astype(str), by_position["accuracy_percent"])
        ax.set_xlabel("Presented gold-answer position")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Accuracy after actively moving the gold option")
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy_by_presented_gold_position.png", dpi=160)
        plt.close(fig)

        hist = per_item["flip_rate"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(hist.index.astype(str), hist.values)
        ax.set_xlabel("Per-item flip rate")
        ax.set_ylabel("Question count")
        ax.set_title("Prediction instability under option permutations")
        fig.tight_layout()
        fig.savefig(out_dir / "flip_rate_distribution.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"[warning] Could not create plots: {exc}")

    report = f"""# Permutation stability report

- Questions: {len(per_item)}
- Prompt evaluations: {len(df)}
- Original-order accuracy: {100 * original_acc:.2f}%
- Shuffled-order accuracy: {100 * shuffled_acc:.2f}%
- Accuracy across all permutations: {100 * all_acc:.2f}%
- Fully stable questions: {100 * summary['fully_stable_item_share']:.2f}%
- Mean flip rate: {100 * summary['mean_flip_rate']:.2f}%
- Questions with at least one prediction flip: {summary['items_with_any_flip']}
- Likelihood fallback rate: {100 * summary['likelihood_fallback_rate']:.2f}%

## Reading the result

A large original-to-shuffled accuracy drop, a low fully-stable share, or strong accuracy differences by `presented_gold_position` indicate position sensitivity. This test maps predictions back to option content, so a harmless letter change caused by moving the same option is not counted as a flip.
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved permutation diagnostics to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
