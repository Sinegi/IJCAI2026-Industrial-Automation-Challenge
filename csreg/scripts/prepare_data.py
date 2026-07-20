#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsiq_csreg.public_data import prepare_public_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FailureSensorIQ and build train-only G*")
    parser.add_argument("--config", default=str(ROOT / "configs" / "a100_40gb.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_dir = args.output_dir or cfg["data"]["prepared_dir"]
    stats = prepare_public_corpus(
        output_dir=output_dir,
        cache_dir=args.cache_dir or cfg["data"].get("hf_cache_dir"),
        permutation_copies=int(cfg["data"]["permutation_copies"]),
        include_graph_synthetic=bool(cfg["data"]["include_graph_synthetic"]),
        graph_source_mode=str(cfg["data"].get("graph_source_mode", "single_only")),
        seed=int(cfg.get("seed", 42)),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["unresolved_single"]:
        print("WARNING: some single-answer rows lack a reliable anchor/direction.")
    if stats["unresolved_multi"]:
        print("WARNING: some multi-answer rows were excluded from graph construction.")


if __name__ == "__main__":
    main()
