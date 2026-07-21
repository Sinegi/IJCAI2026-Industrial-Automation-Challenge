#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsiq_csreg.core import (  # noqa: E402
    CSRegPredictor,
    generate_submission,
    load_public_scenarios,
    seed_everything,
    set_answer_style,
)


def resolve_answer_style(cfg: dict, adapter: str | None) -> str:
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


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Kaggle submission.csv from test.jsonl")
    p.add_argument("--config", default=str(ROOT / "configs" / "a100_40gb.yaml"))
    p.add_argument("--test", default="test.jsonl")
    p.add_argument("--adapter", default=None)
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--tta", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    scenarios = load_public_scenarios(args.test)
    if args.max_rows:
        scenarios = scenarios[: args.max_rows]
    adapter = args.adapter or str(Path(cfg["training"]["output_dir"]) / "final_adapter")
    if isinstance(adapter, str) and adapter.lower() in {"none", "base", "null"}:
        adapter = None
    set_answer_style(resolve_answer_style(cfg, adapter))
    predictor = CSRegPredictor(
        base_model_path=cfg["model"]["base_model"],
        adapter_path=adapter,
        load_in_4bit=True,
        max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
        tta_permutations=int(args.tta or cfg["inference"]["tta_permutations"]),
        inference_batch_size=int(args.batch_size or cfg["inference"]["batch_size"]),
        seed=seed,
    )
    path = generate_submission(predictor, scenarios, args.output)
    df = pd.read_csv(path, dtype=str)
    if list(df.columns) != ["id", "answer"]:
        raise AssertionError("Submission columns must be exactly id,answer")
    if df["id"].duplicated().any():
        raise AssertionError("Duplicate ids in submission")
    if df["answer"].isna().any():
        raise AssertionError("Missing answers in submission")
    print(json.dumps({"rows": len(df), "output": str(path.resolve())}, indent=2))
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
