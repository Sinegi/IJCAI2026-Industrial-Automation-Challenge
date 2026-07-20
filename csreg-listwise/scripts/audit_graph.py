#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Audit the prepared train-only graph bundle")
    p.add_argument("--prepared-dir", default="artifacts/prepared")
    p.add_argument("--output", default="outputs/graph_audit.json")
    args = p.parse_args()

    prepared = Path(args.prepared_dir)
    stats = json.loads((prepared / "prepare_stats.json").read_text(encoding="utf-8"))
    bundle = json.loads((prepared / "graph_bundle.json").read_text(encoding="utf-8"))
    assets = {}
    total_conflicts = 0
    for asset, graph in bundle["graphs"].items():
        votes = graph.get("vote_counts", {})
        conflicts = []
        for pair_key, counts in votes.items():
            positives, total = int(counts[0]), int(counts[1])
            if 0 < positives < total:
                conflicts.append({"pair": pair_key, "positive_votes": positives, "total_votes": total})
        total_conflicts += len(conflicts)
        assets[asset] = {
            "failure_modes": len(graph.get("failure_modes", [])),
            "sensors": len(graph.get("sensors", [])),
            "edges": len(graph.get("edges", [])),
            "observed_nonedges": len(graph.get("observed_nonedges", [])),
            "conflicting_pairs": len(conflicts),
            "conflict_examples": conflicts[:20],
        }
    report = {
        "graph_source_mode": stats.get("graph_source_mode"),
        "graph_source_rows": stats.get("graph_source_rows"),
        "edge_count": stats.get("edge_count"),
        "observed_nonedge_count": stats.get("observed_nonedge_count"),
        "total_conflicting_pairs": total_conflicts,
        "assets": assets,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
