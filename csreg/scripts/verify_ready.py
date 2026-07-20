#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Verify prepared corpus and a saved adapter before full training/evaluation")
    p.add_argument("--prepared-dir", default="artifacts/prepared")
    p.add_argument("--adapter", default=None)
    args = p.parse_args()

    prepared = Path(args.prepared_dir)
    required = [
        prepared / "prepare_stats.json",
        prepared / "graph_bundle.json",
        prepared / "train_rows.jsonl",
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise SystemExit(f"Missing prepared files: {missing}")

    stats = json.loads((prepared / "prepare_stats.json").read_text(encoding="utf-8"))
    checks = {
        "graph_source_mode_is_single_only": stats.get("graph_source_mode") == "single_only",
        "graph_source_rows_is_2667": stats.get("graph_source_rows") == 2667,
        "unresolved_single_is_zero": stats.get("unresolved_single") == 0,
        "ten_assets": len(stats.get("assets", [])) == 10,
        "nonempty_graph": stats.get("edge_count", 0) > 0,
        "nonempty_train_rows": stats.get("training_rows_after_augmentation", 0) > 0,
    }
    print(json.dumps({"stats": stats, "checks": checks}, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit("Prepared corpus failed one or more safety checks. Re-run prepare_data.py with the v3 config.")

    if args.adapter:
        adapter = Path(args.adapter)
        adapter_required = [
            adapter / "adapter_config.json",
            adapter / "csreg_projection.pt",
            adapter / "csreg_config.json",
        ]
        adapter_missing = [str(x) for x in adapter_required if not x.exists()]
        if adapter_missing:
            raise SystemExit(f"Adapter is incomplete: {adapter_missing}")
        print(json.dumps({"adapter": str(adapter.resolve()), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
