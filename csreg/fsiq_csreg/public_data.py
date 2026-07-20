from __future__ import annotations

import json
import re
from collections import Counter, OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .core import (
    EntityLexicon,
    NormalizedMCQ,
    assign_anchors,
    build_entity_lexicon,
    build_train_graphs,
    canonical_asset,
    canonical_entity,
    make_training_rows,
    normalize_record,
    save_graph_bundle,
)

# Canonical names used by the public benchmark. Aliases are only used to infer
# missing metadata in the multi-answer subset; they are never used at test time.
ASSET_ALIASES: dict[str, tuple[str, ...]] = {
    "aero gas turbine": ("aero gas turbine", "aeroderivative gas turbine"),
    "compressor": ("compressor", "reciprocating compressor"),
    "electric generator": ("electric generator", "generator"),
    "electric motor": ("electric motor",),
    "fan": ("fan",),
    "industrial gas turbine": ("industrial gas turbine",),
    "power transformer": ("power transformer", "transformer"),
    "pump": ("pump", "centrifugal pump"),
    "reciprocating ic engine": (
        "reciprocating ic engine",
        "reciprocating internal combustion engine",
        "internal combustion engine",
        "ic engine",
    ),
    "steam turbine": ("steam turbine",),
}

NEGATION_WORDS = re.compile(
    r"\b(?:not|exclude|excluded|irrelevant|unrelated|least|except|should\s+not|cannot)\b",
    re.IGNORECASE,
)


def infer_asset_from_text(text: str, known_assets: Sequence[str] | None = None) -> str:
    q = canonical_entity(text)
    candidates: list[tuple[int, str]] = []
    allowed = set(canonical_asset(a) for a in known_assets) if known_assets else None
    for canonical, aliases in ASSET_ALIASES.items():
        if allowed is not None and canonical not in allowed:
            continue
        for alias in aliases:
            alias_norm = canonical_entity(alias)
            if re.search(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", q):
                candidates.append((len(alias_norm), canonical))
    if candidates:
        return max(candidates)[1]
    if known_assets:
        for asset in sorted((canonical_asset(a) for a in known_assets), key=len, reverse=True):
            if asset in q:
                return asset
    return "unknown"


def infer_relation_metadata(question: str) -> tuple[str, str, str]:
    q = question.lower()
    polarity = "negative" if NEGATION_WORDS.search(q) else "positive"
    if re.search(r"\b(?:which|what)\s+(?:of\s+the\s+(?:following|choices)\s+)?sensors?\b", q) or (
        "sensor" in q and ("failure" in q or "fault" in q) and q.find("sensor") < q.find("failure")
    ):
        direction = "fm2sensor"
    elif re.search(r"\b(?:which|what)\s+(?:of\s+the\s+(?:following|choices)\s+)?(?:failure|fault)(?:\s+(?:modes?|events?))?\b", q):
        direction = "sensor2fm"
    elif "abnormal reading" in q and "sensor" in q:
        direction = "sensor2fm"
    else:
        direction = "unknown"

    if direction == "fm2sensor":
        relevancy = (
            "irrelevant_sensor_for_failure_mode" if polarity == "negative"
            else "relevant_sensors_for_failure_mode"
        )
    elif direction == "sensor2fm":
        relevancy = (
            "irrelevant_failure_modes_for_sensor" if polarity == "negative"
            else "relevant_failure_modes_for_sensor"
        )
    else:
        relevancy = ""
    question_type = "mcp1_negative" if polarity == "negative" else "mcp1_positive"
    return direction, polarity, relevancy or question_type


def enrich_public_row(raw: Mapping[str, Any], known_assets: Sequence[str]) -> dict[str, Any]:
    row = dict(raw)
    question = str(row.get("question", ""))
    if not row.get("asset_name") and not row.get("asset_class"):
        row["asset_name"] = infer_asset_from_text(question, known_assets)
    if not row.get("relevancy"):
        direction, polarity, rel = infer_relation_metadata(question)
        row["relevancy"] = rel
        row["question_type"] = (
            "mcp1_negative" if polarity == "negative" else "mcp1_positive"
        )
    return row


def load_public_rows(cache_dir: str | None = None) -> tuple[list[NormalizedMCQ], list[NormalizedMCQ], EntityLexicon]:
    from datasets import load_dataset

    single_ds = load_dataset(
        "ibm-research/FailureSensorIQ",
        "single_true_multi_choice_qa",
        split="train",
        cache_dir=cache_dir,
    )
    single = [normalize_record(dict(row), i) for i, row in enumerate(single_ds)]
    known_assets = sorted({row.asset for row in single if row.asset != "unknown"})

    # Build the first lexicon from option types; anchor assignment then expands it.
    lexicon = build_entity_lexicon(single)
    assign_anchors(single, lexicon)

    multi_ds = load_dataset(
        "ibm-research/FailureSensorIQ",
        "multi_true_multi_choice_qa",
        split="train",
        cache_dir=cache_dir,
    )
    multi: list[NormalizedMCQ] = []
    for i, raw in enumerate(multi_ds):
        enriched = enrich_public_row(dict(raw), known_assets)
        row = normalize_record(enriched, i)
        multi.append(row)

    # The single subset gives canonical entity vocabularies. Rebuild and then
    # extract anchors from both subsets using longest verbatim match.
    lexicon = build_entity_lexicon(single + multi)
    assign_anchors(single, lexicon)
    assign_anchors(multi, lexicon)
    # A second pass helps when an anchor discovered in pass one appears in later rows.
    assign_anchors(single, lexicon)
    assign_anchors(multi, lexicon)
    return single, multi, lexicon


def row_to_json(row: NormalizedMCQ) -> dict[str, Any]:
    payload = asdict(row)
    payload["options"] = dict(row.options)
    return payload


def write_rows(path: str | Path, rows: Iterable[NormalizedMCQ]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row_to_json(row), ensure_ascii=False) + "\n")


def prepare_public_corpus(
    output_dir: str | Path,
    cache_dir: str | None = None,
    permutation_copies: int = 1,
    include_graph_synthetic: bool = True,
    graph_source_mode: str = "single_only",
    seed: int = 42,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single, multi, lexicon = load_public_rows(cache_dir=cache_dir)
    eligible_single = [
        row for row in single
        if row.is_labeled and row.anchor and row.direction in {"fm2sensor", "sensor2fm"}
    ]
    eligible_multi = [
        row for row in multi
        if row.is_labeled and row.anchor and row.direction in {"fm2sensor", "sensor2fm"}
        and row.asset != "unknown"
    ]
    mode = str(graph_source_mode).strip().lower()
    if mode == "single_only":
        # The single-answer subset contains explicit asset/relevancy metadata and
        # is sufficient to reconstruct the canonical FMEA graph.  The public
        # multi-answer subset omits those metadata fields, so heuristic recovery
        # can create spurious edges.  Keep it out of G* by default.
        graph_source = eligible_single
    elif mode in {"single_plus_multi", "all"}:
        graph_source = eligible_single + eligible_multi
    else:
        raise ValueError(
            f"Unsupported graph_source_mode={graph_source_mode!r}; "
            "use 'single_only' or 'single_plus_multi'"
        )
    graphs = build_train_graphs(graph_source)
    train_rows = make_training_rows(
        single_rows=[row for row in single if row.anchor and row.is_single_answer],
        graphs=graphs,
        permutation_copies=permutation_copies,
        include_graph_synthetic=include_graph_synthetic,
        seed=seed,
    )

    write_rows(output_dir / "public_single_normalized.jsonl", single)
    write_rows(output_dir / "public_multi_normalized.jsonl", multi)
    write_rows(output_dir / "train_rows.jsonl", train_rows)
    save_graph_bundle(output_dir / "graph_bundle.json", graphs, lexicon)

    unresolved_single = sum(not row.anchor or row.direction == "unknown" for row in single)
    unresolved_multi = sum(not row.anchor or row.direction == "unknown" or row.asset == "unknown" for row in multi)
    stats = {
        "single_rows": len(single),
        "multi_rows": len(multi),
        "graph_source_mode": mode,
        "graph_source_rows": len(graph_source),
        "eligible_single_graph_rows": len(eligible_single),
        "eligible_multi_graph_rows": len(eligible_multi),
        "training_rows_after_augmentation": len(train_rows),
        "assets": sorted(graphs),
        "edge_count": int(sum(len(g.edges) for g in graphs.values())),
        "observed_nonedge_count": int(sum(len(g.observed_nonedges) for g in graphs.values())),
        "unresolved_single": int(unresolved_single),
        "unresolved_multi": int(unresolved_multi),
        "asset_counts_single": dict(Counter(row.asset for row in single)),
        "direction_counts_single": dict(Counter(row.direction for row in single)),
        "polarity_counts_single": dict(Counter(row.polarity for row in single)),
    }
    (output_dir / "prepare_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def read_prepared_rows(path: str | Path) -> list[NormalizedMCQ]:
    rows: list[NormalizedMCQ] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            rows.append(
                NormalizedMCQ(
                    id=str(p["id"]),
                    question=str(p["question"]),
                    options=OrderedDict((str(k), str(v)) for k, v in p["options"].items()),
                    passage=str(p.get("passage", "")),
                    asset=str(p.get("asset", "unknown")),
                    relevancy=str(p.get("relevancy", "")),
                    question_type=str(p.get("question_type", "")),
                    direction=str(p.get("direction", "unknown")),
                    polarity=str(p.get("polarity", "positive")),
                    anchor=str(p.get("anchor", "")),
                    correct_labels=[str(x) for x in p.get("correct_labels", [])],
                    reasoning=str(p.get("reasoning", "")),
                    metadata=dict(p.get("metadata", {})),
                )
            )
    return rows
