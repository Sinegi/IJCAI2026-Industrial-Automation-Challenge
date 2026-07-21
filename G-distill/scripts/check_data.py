#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsiq_csreg.core import load_normalized_records, read_json_records  # noqa: E402


def summarize(path: str) -> dict:
    raw = read_json_records(path)
    rows = load_normalized_records(path)
    return {
        "path": str(Path(path).resolve()),
        "rows": len(rows),
        "raw_keys_first_row": sorted(raw[0].keys()) if raw else [],
        "labeled_rows": sum(r.is_labeled for r in rows),
        "single_answer_rows": sum(r.is_single_answer for r in rows),
        "assets": dict(Counter(r.asset for r in rows)),
        "directions": dict(Counter(r.direction for r in rows)),
        "polarities": dict(Counter(r.polarity for r in rows)),
        "n_options": dict(Counter(len(r.options) for r in rows)),
        "missing_anchor": sum(not r.anchor for r in rows),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val", default="val.jsonl")
    p.add_argument("--test", default="test.jsonl")
    p.add_argument("--output", default="outputs/schema_report.json")
    args = p.parse_args()
    report = {"val": summarize(args.val), "test": summarize(args.test)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["val"]["labeled_rows"] == 0:
        raise SystemExit("val.jsonl has no parseable labels")
    if report["test"]["rows"] == 0:
        raise SystemExit("test.jsonl is empty")


if __name__ == "__main__":
    main()
