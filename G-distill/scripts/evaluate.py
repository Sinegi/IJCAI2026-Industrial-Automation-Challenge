#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsiq_csreg.core import (  # noqa: E402
    CSRegPredictor,
    load_normalized_records,
    seed_everything,
    set_answer_style,
)


def resolve_answer_style(cfg: dict, adapter: str | None) -> str:
    """Prefer the answer_style baked into the trained adapter; fall back to config."""
    if adapter:
        cfg_path = Path(adapter) / "csreg_config.json"
        if cfg_path.exists():
            try:
                saved = json.loads(cfg_path.read_text(encoding="utf-8"))
                if saved.get("answer_style"):
                    return str(saved["answer_style"])
            except (json.JSONDecodeError, OSError):
                pass
    return str((cfg.get("listwise") or {}).get("answer_style", "label"))


def bootstrap_ci(values: np.ndarray, seed: int = 42, n_boot: int = 2000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the adapter on competition val.jsonl")
    p.add_argument("--config", default=str(ROOT / "configs" / "a100_40gb.yaml"))
    p.add_argument("--val", default="val.jsonl")
    p.add_argument("--adapter", default=None)
    p.add_argument("--output-dir", default="outputs/validation")
    p.add_argument("--tta", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--simulate-test",
        action="store_true",
        help=(
            "Build the prompts exactly the way submit.py does for test.jsonl "
            "(drop metadata/answer, empty EntityLexicon) so asset/anchor/direction "
            "are unavailable. Scoring and grouping still use the true labels."
        ),
    )
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    rows = load_normalized_records(args.val)
    # Rows actually fed to the predictor. Normally identical to `rows`; with
    # --simulate-test they are rebuilt through the test-time path so the prompt
    # lacks asset/anchor/direction, matching what submit.py really sees.
    prompt_rows = rows
    rows = [r for r in rows if r.is_single_answer]
    if args.max_rows:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("No labeled single-answer rows found in val.jsonl")

    if args.simulate_test:
        import json as _json

        from fsiq_csreg.core import (  # noqa: E402
            EntityLexicon,
            load_public_scenarios,
            normalize_scenario,
            read_json_records,
        )

        stripped_path = Path(args.output_dir) / "_val_as_test.jsonl"
        stripped_path.parent.mkdir(parents=True, exist_ok=True)
        keep_ids = {str(r.id) for r in rows}
        with stripped_path.open("w", encoding="utf-8") as handle:
            for index, raw in enumerate(read_json_records(args.val)):
                rid = str(raw.get("id", raw.get("scenario_id", index)))
                if rid not in keep_ids:
                    continue
                lean = {k: v for k, v in raw.items() if k not in ("metadata", "answer")}
                lean["id"] = rid
                handle.write(_json.dumps(lean, ensure_ascii=False) + "\n")
        empty_lexicon = EntityLexicon()  # exactly what CSRegPredictor uses for test
        by_id = {
            str(s.id): normalize_scenario(s, empty_lexicon)
            for s in load_public_scenarios(stripped_path)
        }
        missing = [r.id for r in rows if str(r.id) not in by_id]
        if missing:
            raise RuntimeError(f"simulate-test lost {len(missing)} rows (e.g. {missing[:3]})")
        prompt_rows = [by_id[str(r.id)] for r in rows]
        unknown_asset = sum(1 for r in prompt_rows if r.asset == "unknown")
        no_anchor = sum(1 for r in prompt_rows if not r.anchor)
        unknown_dir = sum(1 for r in prompt_rows if r.direction not in ("fm2sensor", "sensor2fm"))
        print(
            f"[simulate-test] prompts rebuilt via the test path: "
            f"asset=unknown {unknown_asset}/{len(prompt_rows)}, "
            f"anchor missing {no_anchor}/{len(prompt_rows)}, "
            f"direction unknown {unknown_dir}/{len(prompt_rows)}"
        )
    else:
        prompt_rows = rows

    adapter = args.adapter or str(Path(cfg["training"]["output_dir"]) / "final_adapter")
    if isinstance(adapter, str) and adapter.lower() in {"none", "base", "null"}:
        adapter = None
    answer_style = resolve_answer_style(cfg, adapter)
    set_answer_style(answer_style)
    print(f"answer_style={answer_style}")
    predictor = CSRegPredictor(
        base_model_path=cfg["model"]["base_model"],
        adapter_path=adapter,
        load_in_4bit=True,
        max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
        tta_permutations=int(args.tta or cfg["inference"]["tta_permutations"]),
        inference_batch_size=int(args.batch_size or cfg["inference"]["batch_size"]),
        seed=seed,
    )
    # Predict on prompt_rows (test-shaped when --simulate-test); score against the
    # true rows, whose options are in the same order so option letters still match.
    outputs = predictor.predict_rows(prompt_rows)

    records = []
    for row, out in zip(rows, outputs):
        pred = out["answer"]
        gold = row.answer_label
        records.append(
            {
                "id": row.id,
                "gold": gold,
                "prediction": pred,
                "correct": int(pred == gold),
                "asset": row.asset,
                "family": row.metadata.get("family", row.relevancy or row.question_type),
                "direction": row.direction,
                "polarity": row.polarity,
                "anchor": row.anchor,
                "n_options": len(row.options),
            }
        )
    df = pd.DataFrame(records)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "val_predictions.csv", index=False)

    accuracy = float(df["correct"].mean())
    ci_low, ci_high = bootstrap_ci(df["correct"].to_numpy(), seed=seed)
    by_asset = df.groupby("asset")["correct"].agg(["count", "mean"]).sort_values("mean")
    by_family = df.groupby("family")["correct"].agg(["count", "mean"]).sort_values("mean")
    by_asset.to_csv(out_dir / "metrics_by_asset.csv")
    by_family.to_csv(out_dir / "metrics_by_family.csv")

    metrics = {
        "n": int(len(df)),
        "accuracy": accuracy,
        "accuracy_percent": 100.0 * accuracy,
        "bootstrap_95_ci": [ci_low, ci_high],
        "macro_asset_accuracy": float(by_asset["mean"].mean()),
        "macro_family_accuracy": float(by_family["mean"].mean()),
        "tta_permutations": int(args.tta or cfg["inference"]["tta_permutations"]),
        "simulate_test": bool(args.simulate_test),
        "adapter": adapter,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("\nWorst assets:\n", by_asset.head(10).to_string())
    print("\nWorst families:\n", by_family.head(10).to_string())


if __name__ == "__main__":
    main()
