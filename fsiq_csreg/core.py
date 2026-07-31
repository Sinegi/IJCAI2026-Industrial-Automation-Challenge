"""Causal-Structure Regularized QLoRA for FailureSensorIQ / IJCAI 2026 Track 1.

The module is designed for Kaggle notebooks and the IBM AssetOpsBench starter-kit
interface.  It deliberately keeps all graph construction on labeled training rows.
No graph or retrieval lookup is used by the inference predictor.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Container, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def stable_int(text: str, modulo: int = 2**31 - 1) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % modulo


# -----------------------------------------------------------------------------
# Competition / dataset normalization
# -----------------------------------------------------------------------------

PUBLIC_EXTRA_FIELDS = {
    "question_type",
    "passage",
    "question",
    "options",
    "option_ids",
    "correct",
    "type",
    "category",
    "asset_class",
    "asset_name",
    "family",
    "domain",
    "phase",
    "difficulty",
    "relevancy",
    "subject",
    "question_first",
    "text_type",
    "answer",
    "label",
    "correct_answer",
    "reasoning",
    "cot",
    "rationale",
    "messages",
}

OPTION_LINE_RE = re.compile(
    r"(?m)^\s*([A-Z])\s*[\.\):\-]\s*(.+?)(?=\n\s*[A-Z]\s*[\.\):\-]\s*|\Z)",
    re.DOTALL,
)
NEGATIVE_RE = re.compile(
    r"\b(?:not|least|irrelevant|unrelated|incorrect|except|should\s+not|would\s+not|cannot)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AssetOpsScenario:
    """Compatible with the official IBM starter-kit scenario surface."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def options(self) -> Any:
        return self.metadata.get("options")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, **self.metadata}


@dataclass
class NormalizedMCQ:
    id: str
    question: str
    options: OrderedDict[str, str]
    passage: str = ""
    asset: str = "unknown"
    relevancy: str = ""
    question_type: str = ""
    direction: str = "unknown"  # fm2sensor | sensor2fm | unknown
    polarity: str = "positive"  # positive | negative
    anchor: str = ""
    correct_labels: list[str] = field(default_factory=list)
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def correct_mask(self) -> list[bool]:
        correct = set(self.correct_labels)
        return [label in correct for label in self.options]

    @property
    def is_labeled(self) -> bool:
        return bool(self.correct_labels)

    @property
    def is_single_answer(self) -> bool:
        return len(self.correct_labels) == 1

    @property
    def answer_label(self) -> str | None:
        return self.correct_labels[0] if self.is_single_answer else None

    @property
    def prompt_question(self) -> str:
        if self.passage.strip():
            return f"{self.passage.strip()}\n\n{self.question.strip()}"
        return self.question.strip()


@dataclass
class AssetGraph:
    asset: str
    failure_modes: list[str]
    sensors: list[str]
    edges: set[tuple[str, str]]
    observed_nonedges: set[tuple[str, str]]
    vote_counts: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)

    def edge(self, failure_mode: str, sensor: str) -> int:
        return int((canonical_entity(failure_mode), canonical_entity(sensor)) in self.edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "failure_modes": self.failure_modes,
            "sensors": self.sensors,
            "edges": [list(x) for x in sorted(self.edges)],
            "observed_nonedges": [list(x) for x in sorted(self.observed_nonedges)],
            "vote_counts": {
                f"{fm}\t{s}": [pos, total]
                for (fm, s), (pos, total) in sorted(self.vote_counts.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssetGraph":
        votes: dict[tuple[str, str], tuple[int, int]] = {}
        for key, value in payload.get("vote_counts", {}).items():
            fm, sensor = key.split("\t", 1)
            votes[(fm, sensor)] = (int(value[0]), int(value[1]))
        return cls(
            asset=str(payload["asset"]),
            failure_modes=list(payload.get("failure_modes", [])),
            sensors=list(payload.get("sensors", [])),
            edges={tuple(x) for x in payload.get("edges", [])},
            observed_nonedges={tuple(x) for x in payload.get("observed_nonedges", [])},
            vote_counts=votes,
        )


@dataclass
class EntityLexicon:
    failure_modes_by_asset: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    sensors_by_asset: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    @property
    def all_failure_modes(self) -> set[str]:
        return set().union(*self.failure_modes_by_asset.values()) if self.failure_modes_by_asset else set()

    @property
    def all_sensors(self) -> set[str]:
        return set().union(*self.sensors_by_asset.values()) if self.sensors_by_asset else set()

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_modes_by_asset": {
                k: sorted(v) for k, v in sorted(self.failure_modes_by_asset.items())
            },
            "sensors_by_asset": {
                k: sorted(v) for k, v in sorted(self.sensors_by_asset.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EntityLexicon":
        obj = cls()
        for k, values in payload.get("failure_modes_by_asset", {}).items():
            obj.failure_modes_by_asset[k].update(values)
        for k, values in payload.get("sensors_by_asset", {}).items():
            obj.sensors_by_asset[k].update(values)
        return obj


def canonical_entity(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).strip().lower())
    return text.strip(" \t\n\r\"'`.,;:()[]{}")


def canonical_asset(text: Any) -> str:
    value = canonical_entity("unknown" if text is None else str(text))
    return value or "unknown"


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    raw_text = p.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []
    if p.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    payload = json.loads(raw_text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "questions", "examples"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    raise ValueError(f"Unsupported JSON shape in {p}: {type(payload).__name__}")


def _structured_options(raw_options: Any, option_ids: Sequence[Any] | None = None) -> OrderedDict[str, str]:
    if isinstance(raw_options, Mapping):
        return OrderedDict((str(k).strip().upper(), str(v).strip()) for k, v in raw_options.items())
    if isinstance(raw_options, Sequence) and not isinstance(raw_options, (str, bytes)):
        if option_ids and len(option_ids) == len(raw_options):
            labels = [str(x).strip().upper() for x in option_ids]
        else:
            labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[: len(raw_options)]
        return OrderedDict((label, str(value).strip()) for label, value in zip(labels, raw_options))
    return OrderedDict()


def parse_text_mcqa(text: str) -> tuple[str, OrderedDict[str, str]]:
    matches = list(OPTION_LINE_RE.finditer(text))
    if not matches:
        return text.strip(), OrderedDict()
    options = OrderedDict()
    for match in matches:
        label = match.group(1).strip().upper()
        value = re.sub(r"\s+", " ", match.group(2).strip())
        options[label] = value
    question_part = text[: matches[0].start()].strip()
    question_part = re.sub(r"(?im)^\s*options\s*:\s*$", "", question_part).strip()
    return question_part, options


def infer_direction(relevancy: str, question: str) -> str:
    rel = canonical_entity(relevancy).replace(" ", "_")
    if "sensor" in rel and "failure_mode" in rel:
        sensor_pos = rel.find("sensor")
        fm_pos = rel.find("failure_mode")
        if sensor_pos < fm_pos and "for_failure_mode" in rel:
            return "fm2sensor"
        if fm_pos < sensor_pos and "for_sensor" in rel:
            return "sensor2fm"
    if any(token in rel for token in (
        "sensors_for_failure", "sensor_for_failure", "failure_to_sensor", "failure2sensor",
    )):
        return "fm2sensor"
    if any(token in rel for token in (
        "failure_modes_for_sensor", "failure_mode_for_sensor", "sensor_to_failure", "sensor2failure",
    )):
        return "sensor2fm"

    q = question.lower()
    asks_sensor = bool(re.search(r"\bwhich\s+(?:of\s+the\s+following\s+)?sensors?\b", q)) or "what sensor" in q
    asks_failure = bool(re.search(r"\bwhich\s+(?:of\s+the\s+following\s+)?failure(?:\s+mode)?s?\b", q)) or "what failure" in q
    if asks_sensor and not asks_failure:
        return "fm2sensor"
    if asks_failure and not asks_sensor:
        return "sensor2fm"
    return "unknown"


def infer_polarity(relevancy: str, question_type: str, question: str) -> str:
    joined = " ".join([relevancy, question_type, question])
    rel_canon = canonical_entity(relevancy).replace(" ", "_")
    qt_canon = canonical_entity(question_type).replace(" ", "_")
    if rel_canon.startswith("irrelevant") or rel_canon.startswith("negation"):
        return "negative"
    if "negative" in qt_canon or "negation" in qt_canon or NEGATIVE_RE.search(joined):
        return "negative"
    return "positive"


def _normalize_correct_labels(raw: Mapping[str, Any], options: OrderedDict[str, str]) -> list[str]:
    labels = list(options)
    correct = raw.get("correct")
    if isinstance(correct, Sequence) and not isinstance(correct, (str, bytes)):
        if len(correct) == len(labels) and all(isinstance(x, (bool, np.bool_)) for x in correct):
            return [label for label, value in zip(labels, correct) if bool(value)]
        values = {str(x).strip().upper() for x in correct}
        return [label for label in labels if label in values]

    for key in ("answer", "correct_answer", "label", "gold", "target"):
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = {str(x).strip().upper() for x in value}
            return [label for label in labels if label in values]
        text = str(value).strip().upper()
        if text in options:
            return [text]
        # Some files store the correct option text instead of its label.
        for label, option_text in options.items():
            if canonical_entity(option_text) == canonical_entity(str(value)):
                return [label]
    return []


def _extract_reasoning(raw: Mapping[str, Any]) -> str:
    for key in ("reasoning", "cot", "rationale", "explanation"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = raw.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            content = str(message.get("content", "")).strip()
            if content:
                think = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL | re.IGNORECASE)
                return think.group(1).strip() if think else content
    return ""


def normalize_record(raw: Mapping[str, Any], index: int = 0) -> NormalizedMCQ:
    metadata: dict[str, Any] = {}
    if isinstance(raw.get("metadata"), Mapping):
        metadata.update(raw["metadata"])
    metadata.update({k: raw[k] for k in PUBLIC_EXTRA_FIELDS if k in raw})

    scenario_id = raw.get("id", raw.get("scenario_id", index))
    passage = str(raw.get("passage", metadata.get("passage", "")) or "").strip()
    question = str(raw.get("question", metadata.get("question", "")) or "").strip()
    options = _structured_options(raw.get("options", metadata.get("options")), raw.get("option_ids"))

    raw_text = str(raw.get("text", raw.get("prompt", "")) or "").strip()
    if not options and raw_text:
        parsed_question, parsed_options = parse_text_mcqa(raw_text)
        options = parsed_options
        if not question:
            question = parsed_question
    elif not question and raw_text:
        question = raw_text

    if not question:
        question = passage
        passage = ""
    if not options:
        raise ValueError(f"Record {scenario_id!r} has no parseable options")

    asset = canonical_asset(
        raw.get(
            "asset_name",
            raw.get(
                "asset_class",
                metadata.get("asset_name", metadata.get("asset_class", "unknown")),
            ),
        )
    )
    family = str(raw.get("family", metadata.get("family", "")) or "")
    relevancy = str(raw.get("relevancy", metadata.get("relevancy", family)) or family)
    question_type = str(
        raw.get("question_type", metadata.get("question_type", raw.get("type", family))) or family
    )
    direction = infer_direction(relevancy, question)
    polarity = infer_polarity(relevancy, question_type, question)
    correct_labels = _normalize_correct_labels(raw, options)

    return NormalizedMCQ(
        id=str(scenario_id),
        question=question,
        options=options,
        passage=passage,
        asset=asset,
        relevancy=relevancy,
        question_type=question_type,
        direction=direction,
        polarity=polarity,
        anchor=canonical_entity(raw.get("anchor", metadata.get("anchor", "")) or ""),
        correct_labels=correct_labels,
        reasoning=_extract_reasoning(raw),
        metadata=metadata,
    )


def load_normalized_records(path: str | Path) -> list[NormalizedMCQ]:
    return [normalize_record(raw, i) for i, raw in enumerate(read_json_records(path))]


def load_hf_failuresensoriq(
    include_multi: bool = True,
    cache_dir: str | None = None,
) -> tuple[list[NormalizedMCQ], list[NormalizedMCQ]]:
    """Load IBM's official public FailureSensorIQ subsets from Hugging Face."""
    from datasets import load_dataset

    single_ds = load_dataset(
        "ibm-research/FailureSensorIQ",
        "single_true_multi_choice_qa",
        split="train",
        cache_dir=cache_dir,
    )
    single = [normalize_record(row, i) for i, row in enumerate(single_ds)]
    multi: list[NormalizedMCQ] = []
    if include_multi:
        multi_ds = load_dataset(
            "ibm-research/FailureSensorIQ",
            "multi_true_multi_choice_qa",
            split="train",
            cache_dir=cache_dir,
        )
        multi = [normalize_record(row, i) for i, row in enumerate(multi_ds)]
    return single, multi


def find_candidate_data_files(root: str | Path = "/kaggle/input") -> dict[str, list[Path]]:
    root = Path(root)
    groups: dict[str, list[Path]] = defaultdict(list)
    if not root.exists():
        return groups
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".parquet"}:
            continue
        name = path.name.lower()
        if "test" in name and "question" in name:
            groups["competition_test"].append(path)
        elif "val" in name or "validation" in name:
            groups["competition_val"].append(path)
        elif "multi" in name and "answer" in name:
            groups["fsiq_multi"].append(path)
        elif name in {"all.jsonl", "single_answer.json", "single_answer.jsonl"} or "single" in name:
            groups["fsiq_single"].append(path)
        elif "cot" in name or "prepared_mcp1" in name:
            groups["cot"].append(path)
        elif "perturb" in name or "perteval" in name:
            groups["perturbed"].append(path)
    for key in groups:
        groups[key] = sorted(groups[key])
    return groups


# -----------------------------------------------------------------------------
# Entity extraction and train-only graph construction
# -----------------------------------------------------------------------------

def build_entity_lexicon(rows: Sequence[NormalizedMCQ]) -> EntityLexicon:
    lexicon = EntityLexicon()
    for row in rows:
        option_values = {canonical_entity(value) for value in row.options.values() if canonical_entity(value)}
        if row.direction == "fm2sensor":
            lexicon.sensors_by_asset[row.asset].update(option_values)
        elif row.direction == "sensor2fm":
            lexicon.failure_modes_by_asset[row.asset].update(option_values)
    return lexicon


def _longest_entity_match(text: str, candidates: Iterable[str]) -> str:
    lower = text.lower()
    matches: list[str] = []
    for candidate in candidates:
        candidate = canonical_entity(candidate)
        if not candidate:
            continue
        # Canonical strings often contain parentheses or slashes, so plain substring
        # matching is more reliable than a strict word-boundary expression.
        if candidate in lower:
            matches.append(candidate)
    return max(matches, key=len) if matches else ""


def extract_anchor(row: NormalizedMCQ, lexicon: EntityLexicon) -> str:
    for key in ("anchor", "anchor_entity"):
        value = row.metadata.get(key)
        if value:
            return canonical_entity(value)
    if row.direction == "fm2sensor":
        direct = row.metadata.get("failure_mode")
        if direct:
            return canonical_entity(direct)
        candidates = lexicon.failure_modes_by_asset.get(row.asset, set()) or lexicon.all_failure_modes
    elif row.direction == "sensor2fm":
        direct = row.metadata.get("sensor")
        if direct:
            return canonical_entity(direct)
        candidates = lexicon.sensors_by_asset.get(row.asset, set()) or lexicon.all_sensors
    else:
        return ""

    search_text = f"{row.passage}\n{row.question}"
    matched = _longest_entity_match(search_text, candidates)
    if matched:
        return matched

    # Conservative direction-specific fallbacks for common public templates.
    if row.direction == "fm2sensor":
        patterns = (
            r"(?:failure mode|failure|fault)\s+[\"']?([^\"'?.,;:]+)",
            r"[\"']([^\"']+)[\"']",
        )
    else:
        patterns = (
            r"(?:sensor|measurement)\s+[\"']?([^\"'?.,;:]+)",
            r"[\"']([^\"']+)[\"']",
        )
    for pattern in patterns:
        match = re.search(pattern, row.question, re.IGNORECASE)
        if match:
            candidate = canonical_entity(match.group(1))
            if candidate and 1 <= len(candidate.split()) <= 12:
                return candidate
    return ""


def assign_anchors(rows: Sequence[NormalizedMCQ], lexicon: EntityLexicon) -> None:
    for row in rows:
        if not row.anchor:
            row.anchor = extract_anchor(row, lexicon)
        if row.anchor:
            if row.direction == "fm2sensor":
                lexicon.failure_modes_by_asset[row.asset].add(row.anchor)
            elif row.direction == "sensor2fm":
                lexicon.sensors_by_asset[row.asset].add(row.anchor)


def pair_from_row_option(row: NormalizedMCQ, option_text: str) -> tuple[str, str] | None:
    anchor = canonical_entity(row.anchor)
    option = canonical_entity(option_text)
    if not anchor or not option:
        return None
    if row.direction == "fm2sensor":
        return anchor, option
    if row.direction == "sensor2fm":
        return option, anchor
    return None


def option_edge_target(row: NormalizedMCQ, label: str) -> int | None:
    if not row.is_labeled:
        return None
    is_correct = label in set(row.correct_labels)
    return int(is_correct if row.polarity == "positive" else not is_correct)


def build_train_graphs(
    rows: Sequence[NormalizedMCQ],
    min_vote_fraction: float = 0.5,
) -> dict[str, AssetGraph]:
    """Build asset-specific G* strictly from labeled training rows."""
    votes: dict[str, dict[tuple[str, str], list[int]]] = defaultdict(lambda: defaultdict(list))
    failure_modes: dict[str, set[str]] = defaultdict(set)
    sensors: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        if not row.is_labeled or row.direction not in {"fm2sensor", "sensor2fm"} or not row.anchor:
            continue
        for label, option_text in row.options.items():
            pair = pair_from_row_option(row, option_text)
            target = option_edge_target(row, label)
            if pair is None or target is None:
                continue
            fm, sensor = pair
            failure_modes[row.asset].add(fm)
            sensors[row.asset].add(sensor)
            votes[row.asset][pair].append(target)

    graphs: dict[str, AssetGraph] = {}
    for asset in sorted(set(failure_modes) | set(sensors)):
        edges: set[tuple[str, str]] = set()
        nonedges: set[tuple[str, str]] = set()
        vote_counts: dict[tuple[str, str], tuple[int, int]] = {}
        for pair, values in votes[asset].items():
            positives = int(sum(values))
            total = len(values)
            vote_counts[pair] = (positives, total)
            if positives / max(total, 1) >= min_vote_fraction:
                edges.add(pair)
            else:
                nonedges.add(pair)
        graphs[asset] = AssetGraph(
            asset=asset,
            failure_modes=sorted(failure_modes[asset]),
            sensors=sorted(sensors[asset]),
            edges=edges,
            observed_nonedges=nonedges,
            vote_counts=vote_counts,
        )
    return graphs


def attach_edge_targets(row: NormalizedMCQ, graphs: Mapping[str, AssetGraph]) -> list[int]:
    graph = graphs.get(row.asset)
    targets: list[int] = []
    for option_text in row.options.values():
        pair = pair_from_row_option(row, option_text)
        if graph is None or pair is None:
            targets.append(-1)
        else:
            targets.append(int(pair in graph.edges))
    return targets


def save_graph_bundle(
    path: str | Path,
    graphs: Mapping[str, AssetGraph],
    lexicon: EntityLexicon,
) -> None:
    payload = {
        "graphs": {asset: graph.to_dict() for asset, graph in graphs.items()},
        "lexicon": lexicon.to_dict(),
        "warning": "TRAIN-ONLY artifact. Never rebuild this file from validation/test labels.",
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_graph_bundle(path: str | Path) -> tuple[dict[str, AssetGraph], EntityLexicon]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    graphs = {asset: AssetGraph.from_dict(value) for asset, value in payload["graphs"].items()}
    return graphs, EntityLexicon.from_dict(payload["lexicon"])


# -----------------------------------------------------------------------------
# Leakage-safe splits and graph-based augmentation
# -----------------------------------------------------------------------------

def group_holdout_split(
    rows: Sequence[NormalizedMCQ],
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[NormalizedMCQ], list[NormalizedMCQ]]:
    """Hold out whole (asset, direction, anchor) groups instead of random rows."""
    groups: dict[tuple[str, str, str], list[NormalizedMCQ]] = defaultdict(list)
    ungrouped: list[NormalizedMCQ] = []
    for row in rows:
        if row.anchor and row.direction != "unknown":
            groups[(row.asset, row.direction, row.anchor)].append(row)
        else:
            ungrouped.append(row)

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    target = max(1, int(round(len(keys) * validation_fraction))) if keys else 0
    val_keys = set(keys[:target])

    train: list[NormalizedMCQ] = []
    val: list[NormalizedMCQ] = []
    for key, examples in groups.items():
        (val if key in val_keys else train).extend(examples)
    # Ungrouped rows cannot support structural validation, but remain useful SFT data.
    train.extend(ungrouped)
    return train, val


def _single_answer_template(
    asset: str,
    direction: str,
    polarity: str,
    anchor: str,
) -> str:
    if direction == "fm2sensor":
        if polarity == "positive":
            return (
                f'For the {asset}, which sensor is relevant for monitoring the failure mode "{anchor}"?'
            )
        return (
            f'For the {asset}, which sensor is NOT relevant for monitoring the failure mode "{anchor}"?'
        )
    if polarity == "positive":
        return (
            f'For the {asset}, which failure mode is relevant when the sensor "{anchor}" shows abnormal behavior?'
        )
    return (
        f'For the {asset}, which failure mode is NOT relevant when the sensor "{anchor}" shows abnormal behavior?'
    )


# -----------------------------------------------------------------------------
# NOTA (None-Of-The-Above) support for abstention
# -----------------------------------------------------------------------------

NOTA_CANONICAL_TEXT = "None of the above"
_NOTA_TEXT_KEYS = frozenset(
    canonical_entity(t)
    for t in (
        "None of the above",
        "none of these",
        "none of the options",
        "no correct option",
        "no correct answer",
        "none",
    )
)


def is_nota_text(text: str) -> bool:
    """True if an option string denotes an abstain / None-of-the-above choice."""
    return canonical_entity(text) in _NOTA_TEXT_KEYS


def nota_option_index(row: NormalizedMCQ) -> int:
    """Position of the NOTA option in ``row.options`` (or -1 if none)."""
    for i, text in enumerate(row.options.values()):
        if is_nota_text(text):
            return i
    return -1


def _explicit_nonedges_ranked(
    graph: AssetGraph,
    direction: str,
    anchor: str,
) -> list[tuple[str, float]]:
    """Explicitly observed non-edges of the anchor, ranked by how plausible they look.

    A NOTA item is hard when its options *resemble* real answers: entities that
    share neighbourhoods with the anchor's genuine partners but are themselves
    known not to relate to it. Scoring is therefore the best Jaccard against any
    true neighbour of the anchor. Candidates come only from
    ``observed_nonedges`` — an unobserved pair is not evidence of unrelatedness,
    and asserting one as "definitely wrong" would be false supervision.
    """
    canonical_anchor = canonical_entity(anchor)
    sensors_by_fm, fm_by_sensor = _neighborhoods(graph)
    if direction == "fm2sensor":
        neighbours, true_partners = fm_by_sensor, sensors_by_fm.get(canonical_anchor, set())
    elif direction == "sensor2fm":
        neighbours, true_partners = sensors_by_fm, fm_by_sensor.get(canonical_anchor, set())
    else:
        return []

    partner_neighbourhoods = [neighbours.get(partner, set()) for partner in true_partners]
    scored: list[tuple[str, float]] = []
    for fm, sensor in graph.observed_nonedges:
        if direction == "fm2sensor":
            if fm != canonical_anchor:
                continue
            candidate = sensor
        else:
            if sensor != canonical_anchor:
                continue
            candidate = fm
        own = neighbours.get(candidate, set())
        score = max((_jaccard(own, other) for other in partner_neighbourhoods), default=0.0)
        scored.append((candidate, score))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def _mix_hard_and_random(
    ranked: Sequence[tuple[str, float]],
    count: int,
    hard_fraction: float,
    rng: random.Random,
) -> list[str]:
    """Take the top-ranked share, fill the rest at random from what remains.

    An all-hard option set overfits the model to small topology differences and
    makes a single graph error catastrophic, so difficulty is deliberately mixed.
    """
    if count <= 0 or len(ranked) < count:
        return []
    hard_count = min(count, max(1, round(count * hard_fraction)))
    chosen = [text for text, _ in ranked[:hard_count]]
    remainder = [text for text, _ in ranked[hard_count:]]
    rng.shuffle(remainder)
    chosen += remainder[: count - len(chosen)]
    return chosen if len(chosen) == count else []


def synthesize_nota_rows_from_graph(
    graphs: Mapping[str, AssetGraph],
    options_per_question: int = 5,
    per_anchor: int = 1,
    min_distractors: int = 2,
    hard_fraction: float = 0.5,
    explicit_nonedges_only: bool = True,
    seed: int = 42,
) -> list[NormalizedMCQ]:
    """Create positive MCQ whose correct answer is NOTA using train-only G*.

    Every listed option is a *non-edge* entity of the anchor (genuinely
    unrelated), so "None of the above" is the only correct choice. This supplies
    the missing "all options mismatch -> abstain" signal for both the LM head and
    the listwise abstain logit. NOTA is placed last to match the competition
    format; the abstain mechanism is content-based, not positional.
    """
    rng = random.Random(seed + 777)
    generated: list[NormalizedMCQ] = []

    def make_nota_row(asset: str, direction: str, anchor: str, distractors: list[str], serial: int) -> NormalizedMCQ:
        entities = list(distractors) + [NOTA_CANONICAL_TEXT]  # NOTA last
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[: len(entities)]
        options = OrderedDict(zip(labels, entities))
        return NormalizedMCQ(
            id=f"synthetic_nota::{asset}::{direction}::{stable_int(anchor)}::{serial}",
            question=_single_answer_template(asset, direction, "positive", anchor),
            options=options,
            asset=asset,
            relevancy=(
                "relevant_sensors_for_failure_mode"
                if direction == "fm2sensor"
                else "relevant_failure_modes_for_sensor"
            ),
            question_type="mcp1_nota",
            direction=direction,
            polarity="positive",
            anchor=anchor,
            correct_labels=[labels[-1]],  # NOTA is the gold option
            metadata={"synthetic_nota_from_train_graph": True},
        )

    for asset, graph in graphs.items():
        all_sensors = sorted(graph.sensors)
        all_fm = sorted(graph.failure_modes)
        sensors_by_fm: dict[str, set[str]] = defaultdict(set)
        fm_by_sensor: dict[str, set[str]] = defaultdict(set)
        for fm, sensor in graph.edges:
            sensors_by_fm[fm].add(sensor)
            fm_by_sensor[sensor].add(fm)

        def candidates(direction: str, anchor: str, fallback: list[str]) -> list[tuple[str, float]]:
            if explicit_nonedges_only:
                return _explicit_nonedges_ranked(graph, direction, anchor)
            # Legacy behaviour: any entity not joined by an edge, unranked. Kept so
            # adapters trained before this policy stay reproducible.
            return [(text, 0.0) for text in fallback]

        for fm in graph.failure_modes:
            ranked = candidates(
                "fm2sensor", fm, [s for s in all_sensors if s not in sensors_by_fm.get(fm, set())]
            )
            if len(ranked) < min_distractors:
                continue
            for i in range(per_anchor):
                k = min(options_per_question - 1, len(ranked))
                distractors = _mix_hard_and_random(ranked, k, hard_fraction, rng)
                if not distractors:
                    continue
                generated.append(make_nota_row(asset, "fm2sensor", fm, distractors, i))

        for sensor in graph.sensors:
            ranked = candidates(
                "sensor2fm", sensor, [f for f in all_fm if f not in fm_by_sensor.get(sensor, set())]
            )
            if len(ranked) < min_distractors:
                continue
            for i in range(per_anchor):
                k = min(options_per_question - 1, len(ranked))
                distractors = _mix_hard_and_random(ranked, k, hard_fraction, rng)
                if not distractors:
                    continue
                generated.append(make_nota_row(asset, "sensor2fm", sensor, distractors, i))

    return generated


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _neighborhoods(graph: AssetGraph) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Edge-only adjacency: sensors reachable from a failure mode and vice versa."""
    sensors_by_fm: dict[str, set[str]] = defaultdict(set)
    fm_by_sensor: dict[str, set[str]] = defaultdict(set)
    for fm, sensor in graph.edges:
        sensors_by_fm[fm].add(sensor)
        fm_by_sensor[sensor].add(fm)
    return sensors_by_fm, fm_by_sensor


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _pair_for(direction: str, anchor: str, candidate: str) -> tuple[str, str] | None:
    """Order an (anchor, candidate) pair as the graph's (failure_mode, sensor)."""
    if direction == "fm2sensor":
        return (anchor, candidate)
    if direction == "sensor2fm":
        return (candidate, anchor)
    return None


def graph_hard_distractor_candidates(
    graph: AssetGraph,
    direction: str,
    anchor: str,
    gold_entity: str,
    *,
    want_edges: bool,
    exclude: Container[str] = (),
) -> list[tuple[str, float]]:
    """Graph-verified distractor candidates ranked by hardness.

    Validity comes first and is decided by G* alone: a candidate is admissible
    only if its pair with the anchor is an *explicitly labelled* edge (for
    negative questions, where the gold is the non-edge) or an *explicitly
    observed* non-edge (for positive questions).  Unobserved pairs are never
    used — "no edge in G*" and "known to be unrelated" are different claims, and
    treating the former as a distractor would inject false supervision.

    Only after validity is settled does hardness matter.  Candidates are scored
    by neighbourhood Jaccard against the gold entity, i.e. an entity that shares
    the gold's relations to *other* anchors while genuinely not relating to this
    one.  That is the structural definition of a near miss; lexical similarity is
    deliberately not used, since it would let surface wording override G*.
    """
    canonical_anchor = canonical_entity(anchor)
    canonical_gold = canonical_entity(gold_entity)
    sensors_by_fm, fm_by_sensor = _neighborhoods(graph)
    if direction == "fm2sensor":
        pool, neighbours = graph.sensors, fm_by_sensor
    elif direction == "sensor2fm":
        pool, neighbours = graph.failure_modes, sensors_by_fm
    else:
        return []

    excluded = {canonical_entity(value) for value in exclude}
    excluded.add(canonical_gold)
    gold_neighbourhood = neighbours.get(canonical_gold, set())

    scored: list[tuple[str, float]] = []
    for candidate in pool:
        canonical_candidate = canonical_entity(candidate)
        if canonical_candidate in excluded:
            continue
        pair = _pair_for(direction, canonical_anchor, canonical_candidate)
        if pair is None:
            continue
        admissible = pair in graph.edges if want_edges else pair in graph.observed_nonedges
        if not admissible:
            continue
        scored.append((candidate, _jaccard(neighbours.get(canonical_candidate, set()), gold_neighbourhood)))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def harvest_question_templates(
    rows: Sequence[NormalizedMCQ],
    min_uses: int = 1,
) -> dict[tuple[str, str], list[str]]:
    """Recover the corpus's own question phrasings as reusable templates.

    Synthetic rows written from a handful of hard-coded sentences are a template
    monoculture: the pattern matrix alone would put ~35% of the corpus on FOUR
    phrasings, and run K established that this model does learn the templates it
    is shown (§23-2). The corpus itself supplies 57 phrasings across the four
    (direction, polarity) groups, so the generator samples those instead of
    inventing its own voice.

    A row contributes a template only when its asset and anchor both appear
    literally in the question, so substitution is exact and reversible.
    """
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        if not row.anchor or row.direction not in {"fm2sensor", "sensor2fm"}:
            continue
        if row.metadata.get("relation_fact") or row.metadata.get("pattern_matrix"):
            continue
        question = row.question
        # Longest token first: an anchor containing the asset name must not be
        # half-replaced by the asset pass.
        for token, slot in sorted(
            [(row.anchor, "{anchor}"), (row.asset, "{asset}")], key=lambda item: -len(item[0])
        ):
            replaced = re.sub(re.escape(token), slot, question, flags=re.IGNORECASE)
            if replaced == question:
                question = None
                break
            question = replaced
        if question is None:
            continue
        # Two G* entities are truncated mid-parenthesis ("frequency response
        # analysis (fra"), so slotting them leaves the closing bracket stranded
        # in the template and every OTHER anchor rendered from it would gain a
        # stray ")". The graph is internally consistent -- val and test carry the
        # same spellings -- so the entity names are left alone and the templates
        # they poison are dropped instead.
        if question.count("(") != question.count(")") or question.count('"') % 2:
            continue
        counts[(row.direction, row.polarity)][question] += 1
    return {
        key: sorted(template for template, n in bucket.items() if n >= min_uses)
        for key, bucket in counts.items()
    }


def _render_question(
    templates: Mapping[tuple[str, str], Sequence[str]] | None,
    asset: str,
    direction: str,
    polarity: str,
    anchor: str,
    rng: random.Random,
) -> str:
    """Phrase a synthetic question in one of the corpus's own voices."""
    if templates:
        bucket = templates.get((direction, polarity))
        if bucket:
            return rng.choice(list(bucket)).format(asset=asset, anchor=anchor)
    return _single_answer_template(asset, direction, polarity, anchor)


# -----------------------------------------------------------------------------
# Component-mediated synthesis (V3 card C11)
# -----------------------------------------------------------------------------

# test asks about components ("Which sensor is most useful to detect a bearing
# failure in an electric motor?") and G* has no component entities -- but its
# FAILURE MODE NAMES do: "damaged impeller", "piston ring fault", "power turbine
# damage". Stripping the damage wording leaves the part.
#
# The unsound version of this is asking whether a component EXISTS in an asset:
# "rotor" never appears in compressor's failure modes, yet a screw compressor
# obviously has rotors, and applying that inference to real test items got 3 of 4
# eliminations wrong (§29-5). G* is closed-world for (failure mode, sensor) pairs
# INSIDE an asset; it is not a parts list.
#
# The sound version stays inside that closed world. A component inherits its
# failure modes' relations, and every distractor is drawn from the SAME asset,
# where the non-relation is observed rather than assumed.
_COMPONENT_LEAD = r"(?:damaged|eccentric|worn|broken|loss of|low|excessive|unequal|hogging or sagging)"
_COMPONENT_TRAIL = (
    r"(?:damage[ds]?|wear(?:/ damage)?|fault[s]?|leakage|leak|blockage|blocked|defects"
    r"|deterioration|looseness|distortion|fouled|dirty|holed|stall|condition/ fault|problem"
    r"|ingress/ content|level|quality deterioration)"
)
# Names that survive the strip but denote an event or a condition, not a part.
_NOT_A_COMPONENT = frozenset(
    {"arcing", "external", "moisture", "overheating", "misalignment", "unbalance",
     "mounting", "ignition", "supply", "through"}
)

COMPONENT_PATTERNS = ("component2sensor", "sensor2component")

# NOTA on the component axis. run N showed the pattern matrix's NOTA rows were
# worth +15.5pp on NOTA-is-gold (§31-2) and run L showed what their absence
# teaches: "a listed option always holds" (§28-4). A new question SHAPE without
# NOTA re-teaches that assumption inside that shape, so the component rows carry
# it too. Nothing new is asserted -- a NOTA-gold item simply lists only options
# the same within-asset G* says are unrelated.
COMPONENT_NOTA_MODES = ("none", "nota_gold", "nota_present")


def component_of_failure_mode(failure_mode: str) -> str | None:
    """The part named inside a failure mode, or None if it names no part."""
    text = failure_mode.strip().lower()
    stripped = re.sub(rf"^\s*{_COMPONENT_LEAD}\s+", "", text)
    stripped = re.sub(rf"\s+{_COMPONENT_TRAIL}\s*$", "", stripped)
    stripped = re.sub(r"\s*/.*$", "", stripped).strip()
    stripped = re.sub(r"\s+(fault|damage|wear)$", "", stripped).strip()
    if not stripped or stripped == text or len(stripped) <= 2:
        return None
    if stripped in _NOT_A_COMPONENT:
        return None
    return stripped


def asset_components(graph: AssetGraph) -> dict[str, set[str]]:
    """component -> the asset's failure modes that name it (canonicalised)."""
    out: dict[str, set[str]] = defaultdict(set)
    for failure_mode in graph.failure_modes:
        component = component_of_failure_mode(failure_mode)
        if component:
            out[component].add(canonical_entity(failure_mode))
    return dict(out)


_COMPONENT_TEMPLATES: dict[tuple[str, str], tuple[str, ...]] = {
    ("component2sensor", "positive"): (
        'For the {asset}, which sensor is most useful for detecting a problem with the {component}?',
        'Which sensor would best reveal {component} trouble on a {asset}?',
        'In {asset}, which sensor should be monitored to catch a failing {component}?',
        'A {asset} maintenance plan needs to cover the {component}. Which sensor serves that?',
    ),
    ("component2sensor", "negative"): (
        'For the {asset}, which sensor is NOT useful for detecting a problem with the {component}?',
        'Which sensor would tell you least about {component} trouble on a {asset}?',
        'In {asset}, which sensor is irrelevant to a failing {component}?',
        'Which sensor can be disregarded when assessing the {component} of a {asset}?',
    ),
    ("sensor2component", "positive"): (
        'The sensor "{sensor}" on a {asset} reads abnormally. Which component is most likely implicated?',
        'In {asset}, an abnormal "{sensor}" reading points to trouble in which component?',
        'A {asset} shows an unusual "{sensor}" measurement. Which part should be inspected?',
        'Which component of a {asset} would explain an abnormal "{sensor}" reading?',
    ),
    ("sensor2component", "negative"): (
        'The sensor "{sensor}" on a {asset} reads abnormally. Which component is NOT implicated?',
        'In {asset}, which component would an abnormal "{sensor}" reading NOT point to?',
        'A {asset} shows an unusual "{sensor}" measurement. Which part can be ruled out?',
        'Which component of a {asset} is irrelevant to an abnormal "{sensor}" reading?',
    ),
}


def synthesize_component_rows_from_graph(
    graphs: Mapping[str, AssetGraph],
    option_counts: Sequence[int] = (4, 5, 6),
    patterns: Sequence[str] = COMPONENT_PATTERNS,
    nota_modes: Sequence[str] = COMPONENT_NOTA_MODES,
    per_cell: int = 1,
    hard_fraction: float = 0.5,
    allow_foreign_filler: bool = True,
    max_foreign_fraction: float = 0.4,
    seed: int = 42,
) -> list[NormalizedMCQ]:
    """Component-level questions, derived entirely from within-asset G*.

    A component is relevant to a sensor when ANY failure mode naming it is an
    edge to that sensor; it is a valid distractor only when ALL of them are
    observed non-edges. Nothing crosses an asset boundary, so no claim is made
    about which parts an asset does or does not have.

    Rows carry LM loss only (``metadata['component_row']``): the anchor is a
    component, which is not a G* entity, so option-level edge targets do not
    exist for it.
    """
    rng = random.Random(seed + 24601)
    unknown = [p for p in patterns if p not in COMPONENT_PATTERNS]
    if unknown:
        raise ValueError(f"unknown component patterns: {unknown}")
    counts = sorted({int(c) for c in option_counts if int(c) >= 3})
    if not counts or per_cell < 1:
        return []

    # Foreign filler and hardness both need vocabulary beyond one asset.
    sensor_pool = sorted({s for g in graphs.values() for s in g.sensors})
    global_neighbours: dict[str, set[str]] = defaultdict(set)
    for g in graphs.values():
        for failure, sensor in g.edges:
            global_neighbours[canonical_entity(sensor)].add(canonical_entity(failure))
            global_neighbours[canonical_entity(failure)].add(canonical_entity(sensor))

    generated: list[NormalizedMCQ] = []
    for asset, graph in sorted(graphs.items()):
        components = asset_components(graph)
        if len(components) < 2:
            continue
        sensors_display = {canonical_entity(s): s for s in graph.sensors}
        native_sensors = set(sensors_display)
        sensors_by_fm: dict[str, set[str]] = defaultdict(set)
        fm_by_sensor: dict[str, set[str]] = defaultdict(set)
        for failure, sensor in graph.edges:
            sensors_by_fm[canonical_entity(failure)].add(canonical_entity(sensor))
            fm_by_sensor[canonical_entity(sensor)].add(canonical_entity(failure))
        all_sensors = set(sensors_display)

        unknown_modes = [m for m in nota_modes if m not in COMPONENT_NOTA_MODES]
        if unknown_modes:
            raise ValueError(f"unknown component nota modes: {unknown_modes}")

        for pattern in patterns:
          for nota_mode in nota_modes:
            for polarity in ("positive", "negative"):
                # A NOTA-gold item says "none of these is related", which is the
                # positive question with every listed option drawn from the
                # unrelated side. Asking it negatively ("which is NOT related")
                # while listing only unrelated options would have many answers.
                if nota_mode == "nota_gold" and polarity == "negative":
                    continue
                if pattern == "component2sensor":
                    anchors = sorted(components)
                else:
                    anchors = sorted(graph.sensors)

                for anchor in anchors:
                    if pattern == "component2sensor":
                        related = set().union(
                            *[sensors_by_fm[f] for f in components[anchor]]
                        ) if components[anchor] else set()
                        unrelated = all_sensors - related
                        display = sensors_display
                    else:
                        hit = fm_by_sensor[canonical_entity(anchor)]
                        related = {c for c, fms in components.items() if fms & hit}
                        unrelated = {c for c, fms in components.items() if not (fms & hit)}
                        display = {c: c for c in components}

                    gold_pool = sorted(related if polarity == "positive" else unrelated)
                    wrong_pool = sorted(unrelated if polarity == "positive" else related)
                    if nota_mode == "nota_gold":
                        # Every listed option is unrelated, so NOTA is the answer.
                        gold_pool, wrong_pool = ["__nota__"], sorted(unrelated)
                    if not gold_pool or len(wrong_pool) < 2:
                        continue

                    for n_options in counts:
                        real_slots = n_options - (1 if nota_mode != "none" else 0)
                        needed = real_slots if nota_mode == "nota_gold" else real_slots - 1
                        if needed < 2 or len(wrong_pool) < needed:
                            continue
                        for serial in range(per_cell):
                            gold = gold_pool[serial % len(gold_pool)]
                            # Rank the wrong options the way every other generator
                            # does (§14-2): a distractor that shares the gold's
                            # relations elsewhere is a near miss, one drawn at
                            # random is a giveaway. NOTA-gold items have no gold
                            # entity to rank against, so they stay unranked.
                            reference = None if nota_mode == "nota_gold" else gold
                            if reference is not None:
                                target = global_neighbours.get(canonical_entity(reference), set())
                                ranked = sorted(
                                    ((w, _jaccard(global_neighbours.get(canonical_entity(w), set()), target))
                                     for w in wrong_pool),
                                    key=lambda item: (-item[1], item[0]),
                                )
                                wrong = _mix_hard_and_random(ranked, min(needed, len(ranked)), hard_fraction, rng)
                            else:
                                wrong = list(wrong_pool)
                                rng.shuffle(wrong)
                                wrong = wrong[:needed]
                            entities = [] if nota_mode == "nota_gold" else [display.get(gold, gold)]
                            entities += [display.get(w, w) for w in wrong]

                            # Top up from other assets when this one is too small.
                            # Sound only for component2sensor: a sensor absent from
                            # this asset cannot be the one that watches its part.
                            missing = (n_options - (1 if nota_mode != "none" else 0)) - len(entities)
                            if missing > 0:
                                if (
                                    not allow_foreign_filler
                                    or pattern != "component2sensor"
                                    or polarity != "positive"
                                    or missing > int(max_foreign_fraction * n_options)
                                ):
                                    continue
                                used = {canonical_entity(e) for e in entities}
                                target = global_neighbours.get(
                                    canonical_entity(gold if reference is not None else ""), set()
                                )
                                foreign = sorted(
                                    ((t, _jaccard(global_neighbours.get(canonical_entity(t), set()), target))
                                     for t in sensor_pool
                                     if canonical_entity(t) not in native_sensors
                                     and canonical_entity(t) not in used),
                                    key=lambda item: (-item[1], item[0]),
                                )
                                filler = _mix_hard_and_random(foreign, min(missing, len(foreign)), hard_fraction, rng)
                                if len(filler) < missing:
                                    continue
                                entities += filler
                            if len(set(entities)) != len(entities):
                                continue
                            rng.shuffle(entities)
                            if nota_mode != "none":
                                entities.append(NOTA_CANONICAL_TEXT)  # NOTA last
                            labels = list(ALPHABET)[: len(entities)]
                            options = OrderedDict(zip(labels, entities))
                            if nota_mode == "nota_gold":
                                correct = [labels[-1]]
                            else:
                                gold_display = display.get(gold, gold)
                                correct = [k for k, v in options.items() if v == gold_display]
                            if len(correct) != 1:
                                continue
                            template = rng.choice(_COMPONENT_TEMPLATES[(pattern, polarity)])
                            question = template.format(
                                asset=asset,
                                component=anchor if pattern == "component2sensor" else "",
                                sensor=anchor if pattern == "sensor2component" else "",
                            )
                            generated.append(
                                NormalizedMCQ(
                                    id=(
                                        f"component::{asset}::{pattern}::{polarity}::"
                                        f"{nota_mode}::{n_options}::{stable_int(anchor)}::{serial}"
                                    ),
                                    question=question,
                                    options=options,
                                    asset=asset,
                                    relevancy=(
                                        "relevant_sensors_for_failure_mode"
                                        if pattern == "component2sensor"
                                        else "relevant_failure_modes_for_sensor"
                                    ),
                                    question_type=f"component_{pattern}",
                                    direction=(
                                        "fm2sensor" if pattern == "component2sensor" else "sensor2fm"
                                    ),
                                    polarity=polarity,
                                    anchor=anchor,
                                    correct_labels=correct,
                                    metadata={
                                        "component_row": True,
                                        "component_pattern": pattern,
                                        "component_nota_mode": nota_mode,
                                        "foreign_filler": bool(missing > 0),
                                        "option_count": len(entities),
                                    },
                                )
                            )
    return generated


# -----------------------------------------------------------------------------
# Composite-anchor synthesis (V3 card C8)
# -----------------------------------------------------------------------------

# Every question the corpus and the pattern matrix produce is anchored on ONE
# entity, so the model is never asked to combine two observations. test does ask
# ("Which failure is most likely if BOTH vibration and temperature rise?"), and
# combining is the part that cannot be answered by looking one row up in a
# memorised table -- it needs the intersection or difference of two rows.
#
# AND  : both anchors abnormal   -> gold explains both      (gold in the intersection)
# DIFF : one abnormal, one not   -> gold explains only one  (gold in the difference)
#
# The hard distractors write themselves: for AND, an entity that explains exactly
# one anchor; for DIFF, an entity in the intersection, which would have made the
# second anchor abnormal too.
COMPOSITE_PATTERNS = ("and", "diff")

_COMPOSITE_TEMPLATES: dict[tuple[str, str], tuple[str, ...]] = {
    ("sensor2fm", "and"): (
        'For the {asset}, both the sensor "{a}" and the sensor "{b}" show abnormal readings at the same time. Which failure mode is consistent with both?',
        'In {asset}, abnormal behaviour is observed on "{a}" and on "{b}" together. Which single failure mode accounts for both?',
        'A {asset} reports simultaneous abnormal readings from "{a}" and "{b}". Which failure mode explains both observations?',
        'When "{a}" and "{b}" are both abnormal on a {asset}, which failure mode is indicated by the pair?',
    ),
    ("sensor2fm", "diff"): (
        'For the {asset}, the sensor "{a}" reads abnormally while the sensor "{b}" stays normal. Which failure mode fits this combination?',
        'In {asset}, "{a}" is abnormal but "{b}" shows nothing unusual. Which failure mode is consistent with that?',
        'A {asset} shows an abnormal "{a}" with a normal "{b}". Which failure mode does this point to?',
        'Given an abnormal reading on "{a}" and a normal reading on "{b}" in a {asset}, which failure mode is indicated?',
    ),
    ("fm2sensor", "and"): (
        'For the {asset}, which sensor is relevant for monitoring both the failure mode "{a}" and the failure mode "{b}"?',
        'In {asset}, which single sensor would respond to "{a}" as well as to "{b}"?',
        'A maintenance plan for a {asset} must cover both "{a}" and "{b}". Which sensor serves both?',
        'Which sensor on a {asset} is informative for "{a}" and for "{b}" alike?',
    ),
    ("fm2sensor", "diff"): (
        'For the {asset}, which sensor responds to the failure mode "{a}" but not to the failure mode "{b}"?',
        'In {asset}, which sensor would distinguish "{a}" from "{b}" by responding only to the former?',
        'A {asset} needs a sensor that flags "{a}" while staying quiet for "{b}". Which one is it?',
        'Which sensor on a {asset} is relevant to "{a}" and not to "{b}"?',
    ),
}


def synthesize_composite_anchor_rows_from_graph(
    graphs: Mapping[str, AssetGraph],
    option_counts: Sequence[int] = (4, 5, 6, 7, 8),
    patterns: Sequence[str] = COMPOSITE_PATTERNS,
    per_cell: int = 1,
    hard_fraction: float = 0.7,
    max_pairs_per_asset: int | None = None,
    seed: int = 42,
) -> list[NormalizedMCQ]:
    """Two-anchor questions built from intersections and differences in G*.

    Rows are marked ``metadata['composite_anchor']=True``. Like RFD rows they
    carry LM loss only: the listwise and structural heads score options against a
    SINGLE anchor, and here an option can legitimately be an edge of one anchor
    while still being the wrong answer, so option-level supervision would be
    actively wrong. ``tokenize_training_row`` handles that.
    """
    rng = random.Random(seed + 5150)
    unknown = [p for p in patterns if p not in COMPOSITE_PATTERNS]
    if unknown:
        raise ValueError(f"unknown composite patterns: {unknown}")
    counts = sorted({int(c) for c in option_counts if int(c) >= 3})
    if not counts or per_cell < 1:
        return []

    generated: list[NormalizedMCQ] = []
    for asset, graph in sorted(graphs.items()):
        sensors_by_fm, fm_by_sensor = _neighborhoods(graph)
        display = {canonical_entity(t): t for t in list(graph.sensors) + list(graph.failure_modes)}

        for direction in ("sensor2fm", "fm2sensor"):
            if direction == "sensor2fm":
                adjacency, universe = fm_by_sensor, {canonical_entity(f) for f in graph.failure_modes}
                anchor_names = sorted(graph.sensors)
            else:
                adjacency, universe = sensors_by_fm, {canonical_entity(s) for s in graph.sensors}
                anchor_names = sorted(graph.failure_modes)

            pairs = [
                (a, b)
                for index, a in enumerate(anchor_names)
                for b in anchor_names[index + 1 :]
            ]
            rng.shuffle(pairs)
            if max_pairs_per_asset:
                pairs = pairs[:max_pairs_per_asset]

            for anchor_a, anchor_b in pairs:
                set_a = {canonical_entity(x) for x in adjacency.get(canonical_entity(anchor_a), set())}
                set_b = {canonical_entity(x) for x in adjacency.get(canonical_entity(anchor_b), set())}
                both, only_a = set_a & set_b, set_a - set_b
                neither = universe - set_a - set_b

                for pattern in patterns:
                    if pattern == "and":
                        gold_pool, hard_pool = sorted(both), sorted((set_a | set_b) - both)
                    else:
                        gold_pool, hard_pool = sorted(only_a), sorted(both)
                    easy_pool = sorted(neither)
                    if not gold_pool or not (hard_pool or easy_pool):
                        continue

                    for n_options in counts:
                        for serial in range(per_cell):
                            gold = gold_pool[serial % len(gold_pool)]
                            need = n_options - 1
                            hard = [x for x in hard_pool if x != gold]
                            easy = [x for x in easy_pool if x != gold]
                            rng.shuffle(hard)
                            rng.shuffle(easy)
                            take_hard = min(len(hard), max(1, round(need * hard_fraction)))
                            wrong = hard[:take_hard] + easy[: need - take_hard]
                            if len(wrong) < need:
                                wrong += [x for x in hard[take_hard:] if x not in wrong]
                            if len(wrong) < need:
                                continue  # skip rather than pad with anything unverified
                            entities = [display.get(x, x) for x in [gold] + wrong[:need]]
                            rng.shuffle(entities)
                            labels = list(ALPHABET)[: len(entities)]
                            options = OrderedDict(zip(labels, entities))
                            gold_display = display.get(gold, gold)
                            correct = [k for k, v in options.items() if v == gold_display]
                            if len(correct) != 1:
                                continue
                            template = rng.choice(_COMPOSITE_TEMPLATES[(direction, pattern)])
                            generated.append(
                                NormalizedMCQ(
                                    id=(
                                        f"composite::{asset}::{direction}::{pattern}::"
                                        f"{n_options}::{stable_int(anchor_a + anchor_b)}::{serial}"
                                    ),
                                    question=template.format(asset=asset, a=anchor_a, b=anchor_b),
                                    options=options,
                                    asset=asset,
                                    relevancy=(
                                        "relevant_sensors_for_failure_mode"
                                        if direction == "fm2sensor"
                                        else "relevant_failure_modes_for_sensor"
                                    ),
                                    question_type=f"composite_{pattern}",
                                    direction=direction,
                                    polarity="positive",
                                    anchor=anchor_a,
                                    correct_labels=correct,
                                    metadata={
                                        "composite_anchor": True,
                                        "composite_pattern": pattern,
                                        "anchor_b": anchor_b,
                                        "option_count": len(entities),
                                    },
                                )
                            )
    return generated


# -----------------------------------------------------------------------------
# Pattern-matrix synthesis (V3 card C7)
# -----------------------------------------------------------------------------

# Every pattern the benchmark can pose about a relation table, enumerated. The
# census in §27 found three shapes present in val/test but absent from train:
# NOTA-gold rows (train 0% vs val 22.5% / test 10.0%), option counts 6-8, and
# NOTA-present-but-answerable rows. Growing existing questions cannot reach them
# because a question's option count IS its anchor's degree in the asked
# direction -- the corpus already spends the whole neighbourhood. So the space is
# enumerated instead of grown.
#
# Naming: "pos"/"neg" is the question polarity, "real"/"nota" is what the gold
# is, and "+n" means a NOTA option is present without being the answer.
PATTERN_MATRIX = (
    "pos_real",       # which IS relevant -> gold is an edge
    "neg_real",       # which is NOT relevant -> gold is an observed non-edge
    "pos_nota",       # nothing listed is relevant -> gold is NOTA
    "neg_nota",       # everything listed is relevant -> gold is NOTA
    "pos_real_nota",  # gold is an edge, NOTA present but wrong
    "neg_real_nota",  # gold is a non-edge, NOTA present but wrong
)

# Foreign filler is only sound for POSITIVE questions. A sensor that belongs to
# another asset is genuinely not relevant here, which makes it a valid wrong
# answer to "which IS relevant" -- but it is a second CORRECT answer to "which
# is NOT relevant". Negative patterns therefore stay inside the asset and are
# capped by the anchor's edge degree.
_FOREIGN_SAFE_PATTERNS = frozenset({"pos_real", "pos_nota", "pos_real_nota"})


def synthesize_pattern_matrix_rows_from_graph(
    graphs: Mapping[str, AssetGraph],
    option_counts: Sequence[int] = (4, 5, 6, 7, 8),
    patterns: Sequence[str] = PATTERN_MATRIX,
    per_cell: int = 1,
    hard_fraction: float = 0.5,
    allow_foreign_filler: bool = True,
    max_foreign_fraction: float = 0.5,
    target_nota_share: float | None = None,
    max_rows: int | None = None,
    question_templates: Mapping[tuple[str, str], Sequence[str]] | None = None,
    seed: int = 42,
) -> list[NormalizedMCQ]:
    """Emit every (pattern x option count x anchor x direction) cell G* supports.

    Validity is still decided by G* alone and the gold is never ambiguous:

    * positive gold is an edge, its real distractors are observed non-edges
    * negative gold is an observed non-edge, its real distractors are edges
    * ``pos_nota`` lists only non-edges, so nothing listed is relevant
    * ``neg_nota`` lists only edges, so nothing listed is irrelevant

    ``allow_foreign_filler`` tops a positive item up with entities of the right
    type taken from OTHER assets and absent from this one. They cannot be the
    answer to "which is relevant", so the gold stays unique -- and val builds
    11.8% of its items this way, so refusing to is what capped us below the
    benchmark's own option counts.
    """
    rng = random.Random(seed + 90210)
    unknown = [p for p in patterns if p not in PATTERN_MATRIX]
    if unknown:
        raise ValueError(f"unknown patterns: {unknown}")
    counts = sorted({int(c) for c in option_counts if int(c) >= 2})
    if not counts or per_cell < 1:
        return []

    # Entity vocabulary per type, used only to source foreign filler.
    all_sensors: dict[str, set[str]] = {}
    all_failures: dict[str, set[str]] = {}
    for asset, graph in graphs.items():
        all_sensors[asset] = {canonical_entity(s) for s in graph.sensors}
        all_failures[asset] = {canonical_entity(f) for f in graph.failure_modes}
    sensor_pool = sorted({s for g in graphs.values() for s in g.sensors})
    failure_pool = sorted({f for g in graphs.values() for f in g.failure_modes})

    generated: list[NormalizedMCQ] = []

    # Global adjacency: which partners an entity is joined to ANYWHERE. Used to
    # rank foreign filler, since a foreign entity has no edges inside this asset
    # by construction and local Jaccard would be uniformly zero.
    global_neighbours: dict[str, set[str]] = defaultdict(set)
    for graph in graphs.values():
        for failure, sensor in graph.edges:
            global_neighbours[canonical_entity(sensor)].add(canonical_entity(failure))
            global_neighbours[canonical_entity(failure)].add(canonical_entity(sensor))

    def foreign_options(
        asset: str, direction: str, used: set[str], want: int, gold_text: str | None
    ) -> list[str]:
        """Wrong answers borrowed from another asset, hardest first.

        Picking these at random makes them obviously wrong ("steam leakage" in an
        electric motor question), which teaches "delete the word that does not
        belong here" instead of the relation. Ranking by how much a candidate's
        GLOBAL neighbourhood overlaps the gold's keeps them plausible: a sensor
        that answers to the same failure modes elsewhere reads like a real
        contender. Validity is untouched -- the entity still does not occur in
        this asset, so it still cannot be the answer.
        """
        pool = sensor_pool if direction == "fm2sensor" else failure_pool
        native = (all_sensors if direction == "fm2sensor" else all_failures)[asset]
        options = [
            text
            for text in pool
            if canonical_entity(text) not in native and canonical_entity(text) not in used
        ]
        if not options:
            return []
        if gold_text and hard_fraction > 0.0:
            gold_neighbourhood = global_neighbours.get(canonical_entity(gold_text), set())
            ranked = sorted(
                ((text, _jaccard(global_neighbours.get(canonical_entity(text), set()), gold_neighbourhood))
                 for text in options),
                key=lambda item: (-item[1], item[0]),
            )
            return _mix_hard_and_random(ranked, min(want, len(ranked)), hard_fraction, rng)
        rng.shuffle(options)
        return options[:want]

    for asset, graph in sorted(graphs.items()):
        sensors_by_fm, fm_by_sensor = _neighborhoods(graph)
        anchors = [("fm2sensor", fm) for fm in graph.failure_modes]
        anchors += [("sensor2fm", sensor) for sensor in graph.sensors]

        for direction, anchor in anchors:
            canonical_anchor = canonical_entity(anchor)
            if direction == "fm2sensor":
                edges = sorted(
                    s for f, s in graph.edges if canonical_entity(f) == canonical_anchor
                )
            else:
                edges = sorted(
                    f for f, s in graph.edges if canonical_entity(s) == canonical_anchor
                )
            nonedges_ranked = _explicit_nonedges_ranked(graph, direction, anchor)
            nonedges = [text for text, _ in nonedges_ranked]
            if not edges and not nonedges:
                continue

            for pattern in patterns:
                positive = pattern.startswith("pos")
                gold_is_nota = pattern.endswith("_nota") and "real" not in pattern
                nota_present = pattern.endswith("_nota")
                # Which side supplies the wrong answers.
                wrong_pool = list(nonedges) if positive else list(edges)
                gold_pool = list(edges) if positive else list(nonedges)

                for n_options in counts:
                    for serial in range(per_cell):
                        used: set[str] = {canonical_anchor}
                        entities: list[str] = []
                        gold_text: str | None = None

                        if not gold_is_nota:
                            if not gold_pool:
                                continue
                            gold_text = gold_pool[serial % len(gold_pool)]
                            entities.append(gold_text)
                            used.add(canonical_entity(gold_text))

                        # Real wrong answers, hardest first then random (§14-2).
                        slots = n_options - len(entities) - (1 if nota_present else 0)
                        if slots < 1:
                            continue
                        available = [
                            (text, score)
                            for text, score in (
                                nonedges_ranked
                                if positive
                                else [(text, 0.0) for text in wrong_pool]
                            )
                            if canonical_entity(text) not in used
                        ]
                        wrong = _mix_hard_and_random(available, min(slots, len(available)), hard_fraction, rng)
                        used.update(canonical_entity(text) for text in wrong)
                        entities.extend(wrong)

                        missing = n_options - len(entities) - (1 if nota_present else 0)
                        if missing > 0:
                            allowed = int(max_foreign_fraction * n_options)
                            if (
                                not allow_foreign_filler
                                or pattern not in _FOREIGN_SAFE_PATTERNS
                                or missing > allowed
                            ):
                                continue
                            filler = foreign_options(asset, direction, used, missing, gold_text)
                            if len(filler) < missing:
                                continue
                            used.update(canonical_entity(text) for text in filler)
                            entities.extend(filler)

                        rng.shuffle(entities)
                        if nota_present:
                            entities.append(NOTA_CANONICAL_TEXT)  # NOTA last, per format
                        labels = list(ALPHABET)[: len(entities)]
                        options = OrderedDict(zip(labels, entities))
                        if gold_is_nota:
                            correct = [labels[-1]]
                        else:
                            correct = [
                                label
                                for label, value in options.items()
                                if value == gold_text
                            ]
                            if len(correct) != 1:
                                continue
                        generated.append(
                            NormalizedMCQ(
                                id=(
                                    f"pattern::{asset}::{direction}::{pattern}::"
                                    f"{n_options}::{stable_int(anchor)}::{serial}"
                                ),
                                question=_render_question(
                                    question_templates,
                                    asset,
                                    direction,
                                    "positive" if positive else "negative",
                                    anchor,
                                    rng,
                                ),
                                options=options,
                                asset=asset,
                                relevancy=(
                                    "relevant_sensors_for_failure_mode"
                                    if direction == "fm2sensor"
                                    else "relevant_failure_modes_for_sensor"
                                ),
                                question_type=f"pattern_{pattern}",
                                direction=direction,
                                polarity="positive" if positive else "negative",
                                anchor=anchor,
                                correct_labels=correct,
                                metadata={
                                    "pattern_matrix": True,
                                    "pattern": pattern,
                                    "option_count": len(entities),
                                    "foreign_filler": bool(missing > 0),
                                },
                            )
                        )

    # The full matrix is 68% NOTA-bearing because four of the six patterns carry
    # a NOTA option, but val is 22.5% and test 10.0%. Shipping the raw mix would
    # repeat run H's mistake (§20: NOTA and relational sharpness competed for 8B
    # capacity), so the composition is matched to the benchmark instead.
    if target_nota_share is not None:
        rng_mix = random.Random(seed + 31337)
        with_nota = [row for row in generated if nota_option_index(row) >= 0]
        without = [row for row in generated if nota_option_index(row) < 0]
        share = min(max(float(target_nota_share), 0.0), 1.0)
        if share <= 0.0:
            generated = without
        elif without and share < 1.0:
            keep = min(len(with_nota), int(round(len(without) * share / (1.0 - share))))
            rng_mix.shuffle(with_nota)
            generated = without + with_nota[:keep]
            generated.sort(key=lambda row: row.id)

    if max_rows is not None and len(generated) > max_rows:
        # Stratify by pattern so a cap never silently deletes a whole pattern.
        rng_cap = random.Random(seed + 8675309)
        by_pattern: dict[str, list[NormalizedMCQ]] = defaultdict(list)
        for row in generated:
            by_pattern[row.metadata["pattern"]].append(row)
        quota = max(1, max_rows // max(len(by_pattern), 1))
        capped: list[NormalizedMCQ] = []
        for pattern in sorted(by_pattern):
            bucket = by_pattern[pattern]
            rng_cap.shuffle(bucket)
            capped.extend(bucket[:quota])
        rng_cap.shuffle(capped)
        generated = sorted(capped[:max_rows], key=lambda row: row.id)

    return generated


def augment_option_counts_from_graph(
    rows: Sequence[NormalizedMCQ],
    graphs: Mapping[str, AssetGraph],
    target_counts: Sequence[int] = (6, 7, 8),
    copies_per_row: int = 1,
    hard_fraction: float = 0.5,
    max_reuse_fraction: float = 0.25,
    seed: int = 42,
    adaptive_target: bool = False,
    shrink_fraction: float = 0.0,
) -> list[NormalizedMCQ]:
    """Resize option sets using graph-verified, hardness-ranked distractors.

    Accuracy degrades monotonically with option count (CODE_SPEC 11-1c), yet the
    corpus offers few large-option items.  This produces variants of existing
    rows at the target sizes, keeping the gold and drawing every added option
    from G*.  Difficulty is mixed on purpose: filling a row with maximally hard
    distractors overfits the model to small topology differences and makes a
    single graph error catastrophic, so only ``hard_fraction`` of the additions
    are top-ranked and the rest are random graph-valid entities.

    Rows whose graph cannot supply enough candidates are skipped rather than
    padded, and per-candidate reuse is capped so high-degree hubs do not end up
    in every generated item.
    """
    rng = random.Random(seed + 4242)
    targets = sorted({int(value) for value in target_counts if int(value) >= 2})
    if not targets or copies_per_row < 1:
        return []

    eligible: list[NormalizedMCQ] = []
    for row in rows:
        if not row.is_single_answer or row.direction not in {"fm2sensor", "sensor2fm"}:
            continue
        if not row.anchor or row.asset not in graphs:
            continue
        # NOTA and relation-fact rows carry their own option semantics.
        if row.metadata.get("relation_fact") or nota_option_index(row) >= 0:
            continue
        if row.metadata.get("composite_anchor") or row.metadata.get("component_row"):
            # Resizing a composite item would draw distractors against one anchor
            # only and could hand it a second correct answer. A component row's
            # anchor is not a G* entity at all, so the augmenter cannot rank
            # candidates against it.
            continue
        eligible.append(row)
    if not eligible:
        return []

    # Cap how often one entity may appear across the generated corpus, so a
    # high-degree sensor cannot become a shortcut feature.
    reuse_cap = max(1, int(max_reuse_fraction * len(eligible) * copies_per_row))
    usage: Counter = Counter()
    generated: list[NormalizedMCQ] = []

    for row in eligible:
        graph = graphs[row.asset]
        gold_label = row.answer_label
        assert gold_label is not None
        gold_entity = row.options[gold_label]
        # Positive: gold is an edge, so distractors must be observed non-edges.
        # Negative: gold is the non-edge, so every distractor must be a real edge.
        want_edges = row.polarity == "negative"
        candidates = graph_hard_distractor_candidates(
            graph,
            row.direction,
            row.anchor,
            gold_entity,
            want_edges=want_edges,
            exclude=list(row.options.values()),
        )
        if not candidates:
            continue

        for copy_index in range(copies_per_row):
            current = len(row.options)
            # Prefer growing: the measured weakness is at high option counts.
            grow = [value for value in targets if value > current]
            shrink = [value for value in targets if value < current]
            # ...except for a reserved share of copies. Run REG2's errors put the
            # gold at rank 2 in 25 of 27 cases with a top1-top2 margin of 0.927 --
            # as confident as when it is right. The model already excludes the
            # easy options; what it never trains on is the final two-way call,
            # because growing-only augmentation never produces a small item.
            want_shrink = shrink and rng.random() < shrink_fraction
            choices = (
                shrink
                if want_shrink
                else (grow or [value for value in targets if value != current])
            )
            if not choices:
                continue
            target = rng.choice(choices)

            available = [(text, score) for text, score in candidates if usage[canonical_entity(text)] < reuse_cap]
            if target > current:
                needed = target - current
                if len(available) < needed:
                    if not adaptive_target or not available:
                        continue  # skip rather than pad with unverified options
                    # Adaptive target (CODE_SPEC §26): the corpus questions already
                    # consume most of each anchor's observed non-edges, so demanding a
                    # fixed size drops 45-66% of eligible rows -- and it drops them in
                    # proportion to how sparse the asset's graph is, which routes the
                    # augmentation budget AWAY from the weak assets. Growing as far as
                    # this anchor allows keeps the row instead. Still no padding: every
                    # added option comes from the same validated candidate list.
                    needed = len(available)
                    target = current + needed
                hard_count = min(len(available), max(1, round(needed * hard_fraction)))
                hard = [text for text, _ in available[:hard_count]][:needed]
                remainder = [text for text, _ in available[hard_count:]]
                rng.shuffle(remainder)
                additions = hard + remainder[: needed - len(hard)]
                if len(additions) < needed:
                    continue
                entities = list(row.options.values()) + additions
            else:
                # Shrinking must not make the item trivially easy, so the kept
                # distractors are the ones structurally closest to the gold.
                others = [value for label, value in row.options.items() if label != gold_label]
                # Rank against a candidate list that does NOT exclude this row's
                # own options. `candidates` excludes them by construction, so
                # scoring `others` against it returns 0.0 for every one of them
                # and the "hardest" sort silently degrades to input order --
                # which is exactly the ranking that matters at two options.
                ranking = {
                    canonical_entity(text): score
                    for text, score in graph_hard_distractor_candidates(
                        graph,
                        row.direction,
                        row.anchor,
                        gold_entity,
                        want_edges=want_edges,
                        exclude=[gold_entity],
                    )
                }
                ranked = sorted(
                    others,
                    key=lambda text: -ranking.get(canonical_entity(text), 0.0),
                )
                keep = max(1, target - 1)
                hard_keep = min(keep, max(1, round(keep * hard_fraction)))
                pool = ranked[:hard_keep]
                rest = ranked[hard_keep:]
                rng.shuffle(rest)
                pool += rest[: keep - len(pool)]
                if len(pool) < keep:
                    continue
                entities = [gold_entity] + pool

            rng.shuffle(entities)
            labels = list(ALPHABET)[: len(entities)]
            options = OrderedDict(zip(labels, entities))
            correct = [label for label, value in options.items() if value == gold_entity]
            if len(correct) != 1:
                continue  # a duplicate surface form would make the gold ambiguous
            for value in entities:
                if value != gold_entity:
                    usage[canonical_entity(value)] += 1
            generated.append(
                NormalizedMCQ(
                    **{
                        **asdict(row),
                        "id": f"{row.id}::optaug{target}::{copy_index}",
                        "options": options,
                        "correct_labels": correct,
                        "metadata": {
                            **row.metadata,
                            "option_augmented": True,
                            "option_count": len(entities),
                            "source_row_id": row.id,
                        },
                    }
                )
            )

    return generated


def cross_asset_disagreement_pairs(
    graphs: Mapping[str, AssetGraph],
) -> set[tuple[str, str]]:
    """(failure mode, sensor) pairs whose G* label flips between assets.

    22.3% of pairs occur in more than one asset and 89.4% of those agree, so
    transfer across assets is mostly right -- which is exactly why the model does
    it, and exactly why the 10.6% that disagree are where it breaks. Measured on
    run I: val items whose options contain an AGREEING cross-asset pair score
    97.54%, ones containing a DISAGREEING pair score 86.23%, and ones whose gold
    IS a disagreeing pair score 79.17%.

    The disagreements cluster in the most similar assets (aero vs industrial gas
    turbine, 85.4% agreement), i.e. precisely where the analogy is most tempting.
    """
    labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for graph in graphs.values():
        for failure, sensor in graph.edges:
            labels[(canonical_entity(failure), canonical_entity(sensor))].add("edge")
        for failure, sensor in graph.observed_nonedges:
            labels[(canonical_entity(failure), canonical_entity(sensor))].add("nonedge")
    return {pair for pair, seen in labels.items() if len(seen) > 1}


def synthesize_relation_fact_rows_from_graph(
    graphs: Mapping[str, AssetGraph],
    both_directions: bool = True,
    contrast_repeats: int = 1,
    both_option_orders: bool = False,
    seed: int = 42,
) -> list[NormalizedMCQ]:
    """Dense fact distillation: one compact yes/no item per G* (asset, fm, sensor) pair.

    Closed-book requirement means G* can never be consulted at inference, so the
    relation table must live in the weights. The MCQA rows already expose all
    1342 pairs, but each answer token is ~1 of ~95 generated tokens, so the
    relational fact is heavily diluted by the reasoning template. These rows put
    the fact itself in the supervised target with a short reasoning and a 2-option
    (Yes/No) answer, giving a dense, asset-conditioned gradient per pair.

    Rows are marked ``metadata['relation_fact']=True``; ``tokenize_training_row``
    then disables the listwise/structural option supervision for them (the "Yes"/
    "No" options are not entities, so option-content scoring is meaningless) and
    they contribute LM loss only.

    ``both_option_orders`` emits each cell under BOTH Yes/No orders instead of one
    random order. CODE_SPEC 42: the exhaustive probe found run V holds 96.91% of
    G* but 478 of 2,684 cells (17.8%) flip sign when the two option orders are
    swapped -- the fact is stored, just not strongly enough to outvote slot
    position. One random order per cell cannot teach that away; both orders can.
    Default False so every config written before this parameter existed consumes
    the RNG in exactly the same sequence and reassembles bit-for-bit.
    """
    rng = random.Random(seed + 4242)
    generated: list[NormalizedMCQ] = []
    # Repeat only the pairs the model provably mishandles. This adds no new
    # information -- every copy is the same observed fact -- it just stops the
    # asset-conditioned exceptions from being outvoted by the 89.4% of pairs that
    # do transfer. Emphasis, not invention.
    contrast = cross_asset_disagreement_pairs(graphs) if contrast_repeats > 1 else set()

    def make_fact_row(
        asset: str,
        fm: str,
        sensor: str,
        is_edge: bool,
        direction: str,
        serial: int,
        order: tuple[str, str] | None = None,
    ) -> NormalizedMCQ:
        anchor, other = (fm, sensor) if direction == "fm2sensor" else (sensor, fm)
        if direction == "fm2sensor":
            question = (
                f'For the {asset}, is the sensor "{sensor}" relevant for monitoring '
                f'the failure mode "{fm}"?'
            )
            reasoning = (
                f'In the {asset} FMEA relation, "{sensor}" is '
                f'{"a relevant sensor" if is_edge else "not a relevant sensor"} for "{fm}".'
            )
        else:
            question = (
                f'For the {asset}, is the failure mode "{fm}" indicated when the sensor '
                f'"{sensor}" shows an abnormal reading?'
            )
            reasoning = (
                f'In the {asset} FMEA relation, "{fm}" is '
                f'{"a related failure mode" if is_edge else "not a related failure mode"} for "{sensor}".'
            )
        if order is None:
            choices = ["Yes", "No"]
            rng.shuffle(choices)  # avoid a fixed Yes-is-A positional shortcut
            suffix = ""
        else:
            # Explicit order: no RNG draw, and the id has to carry the order or
            # the two copies of a cell collide on the same key.
            choices = list(order)
            suffix = f"::{choices[0].lower()}first"
        labels = ["A", "B"]
        options = OrderedDict(zip(labels, choices))
        gold_text = "Yes" if is_edge else "No"
        return NormalizedMCQ(
            id=f"relfact::{asset}::{direction}::{stable_int(fm + '|' + sensor)}::{serial}{suffix}",
            question=question,
            options=options,
            asset=asset,
            relevancy=(
                "relevant_sensors_for_failure_mode"
                if direction == "fm2sensor"
                else "relevant_failure_modes_for_sensor"
            ),
            question_type="relation_fact",
            direction=direction,
            polarity="positive",
            anchor=anchor,
            correct_labels=[labels[choices.index(gold_text)]],
            reasoning=reasoning,
            metadata={"relation_fact": True, "asset": asset, "other_entity": other},
        )

    for asset, graph in graphs.items():
        pairs = [(p, True) for p in sorted(graph.edges)] + [
            (p, False) for p in sorted(graph.observed_nonedges)
        ]
        for serial, ((fm, sensor), is_edge) in enumerate(pairs):
            copies = (
                contrast_repeats
                if (canonical_entity(fm), canonical_entity(sensor)) in contrast
                else 1
            )
            orders: tuple[tuple[str, str] | None, ...] = (
                (("Yes", "No"), ("No", "Yes")) if both_option_orders else (None,)
            )
            for copy_index in range(copies):
                for order in orders:
                    generated.append(
                        make_fact_row(
                            asset, fm, sensor, is_edge, "fm2sensor",
                            serial + copy_index * 100003, order,
                        )
                    )
                    if both_directions:
                        generated.append(
                            make_fact_row(
                                asset, fm, sensor, is_edge, "sensor2fm",
                                serial + copy_index * 100003, order,
                            )
                        )

    return generated


def synthesize_answerable_with_nota_rows_from_graph(
    graphs: Mapping[str, AssetGraph],
    options_per_question: int = 5,
    per_anchor: int = 1,
    hard_fraction: float = 0.5,
    explicit_nonedges_only: bool = True,
    seed: int = 42,
) -> list[NormalizedMCQ]:
    """Calibration rows: a NOTA option is present but a REAL edge option is correct.

    Every NOTA-bearing training row otherwise has gold = NOTA, which teaches the
    spurious prior "NOTA option present -> pick NOTA". These rows supply the
    opposite case (answerable *with* a None-of-the-above choice) so the abstain
    gate learns "when a real option clearly matches the anchor, do NOT abstain".
    Improvement-A isolation does not fire here because gold != NOTA, so the real
    relational representation is still trained.
    """
    rng = random.Random(seed + 555)
    generated: list[NormalizedMCQ] = []

    def make_row(asset: str, direction: str, anchor: str, correct: str, distractors: list[str], serial: int) -> NormalizedMCQ:
        reals = [correct] + distractors
        rng.shuffle(reals)
        entities = reals + [NOTA_CANONICAL_TEXT]  # NOTA present but NOT the answer, placed last
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[: len(entities)]
        options = OrderedDict(zip(labels, entities))
        return NormalizedMCQ(
            id=f"synthetic_answerable_nota::{asset}::{direction}::{stable_int(anchor)}::{serial}",
            question=_single_answer_template(asset, direction, "positive", anchor),
            options=options,
            asset=asset,
            relevancy=(
                "relevant_sensors_for_failure_mode"
                if direction == "fm2sensor"
                else "relevant_failure_modes_for_sensor"
            ),
            question_type="mcp1_answerable_nota",
            direction=direction,
            polarity="positive",
            anchor=anchor,
            correct_labels=[labels[reals.index(correct)]],  # gold = the real edge option
            metadata={"synthetic_answerable_with_nota": True},
        )

    for asset, graph in graphs.items():
        all_sensors = sorted(graph.sensors)
        all_fm = sorted(graph.failure_modes)
        sensors_by_fm: dict[str, set[str]] = defaultdict(set)
        fm_by_sensor: dict[str, set[str]] = defaultdict(set)
        for fm, sensor in graph.edges:
            sensors_by_fm[fm].add(sensor)
            fm_by_sensor[sensor].add(fm)

        def build(direction: str, anchor: str, partners: list[str], fallback: list[str]) -> None:
            if not partners:
                return
            rng.shuffle(partners)
            for i, correct in enumerate(partners[:per_anchor]):
                if explicit_nonedges_only:
                    # gold is a real edge here, so hardness is measured against it:
                    # entities sharing the gold's other relations but observed NOT
                    # to relate to this anchor.
                    ranked = graph_hard_distractor_candidates(
                        graph, direction, anchor, correct, want_edges=False
                    )
                else:
                    ranked = [(text, 0.0) for text in fallback]
                k = min(options_per_question - 2, len(ranked))  # leave room for gold + NOTA
                distractors = _mix_hard_and_random(ranked, k, hard_fraction, rng) if k > 0 else []
                if k > 0 and not distractors:
                    continue
                generated.append(make_row(asset, direction, anchor, correct, distractors, i))

        for fm in graph.failure_modes:
            build(
                "fm2sensor",
                fm,
                sorted(sensors_by_fm.get(fm, set())),
                [s for s in all_sensors if s not in sensors_by_fm.get(fm, set())],
            )

        for sensor in graph.sensors:
            build(
                "sensor2fm",
                sensor,
                sorted(fm_by_sensor.get(sensor, set())),
                [f for f in all_fm if f not in fm_by_sensor.get(sensor, set())],
            )

    return generated


def synthesize_single_answer_rows_from_graph(
    graphs: Mapping[str, AssetGraph],
    options_per_question: int = 5,
    positives_per_anchor: int = 2,
    negatives_per_anchor: int = 1,
    seed: int = 42,
) -> list[NormalizedMCQ]:
    """Create direction-flipped, positive, and exclusion MCQ using train-only G*."""
    rng = random.Random(seed)
    generated: list[NormalizedMCQ] = []

    def make_row(
        asset: str,
        direction: str,
        polarity: str,
        anchor: str,
        correct_entity: str,
        distractors: list[str],
        serial: int,
    ) -> NormalizedMCQ:
        entities = [correct_entity] + distractors
        rng.shuffle(entities)
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[: len(entities)]
        options = OrderedDict(zip(labels, entities))
        answer = labels[entities.index(correct_entity)]
        return NormalizedMCQ(
            id=f"synthetic::{asset}::{direction}::{polarity}::{stable_int(anchor)}::{serial}",
            question=_single_answer_template(asset, direction, polarity, anchor),
            options=options,
            asset=asset,
            relevancy=(
                ("relevant_sensors_for_failure_mode" if direction == "fm2sensor" else "relevant_failure_modes_for_sensor")
                if polarity == "positive"
                else ("irrelevant_sensors_for_failure_mode" if direction == "fm2sensor" else "irrelevant_failure_modes_for_sensor")
            ),
            question_type="mcp1_positive" if polarity == "positive" else "mcp1_negative",
            direction=direction,
            polarity=polarity,
            anchor=anchor,
            correct_labels=[answer],
            metadata={"synthetic_from_train_graph": True},
        )

    for asset, graph in graphs.items():
        all_fm = set(graph.failure_modes)
        all_sensors = set(graph.sensors)
        sensors_by_fm: dict[str, set[str]] = defaultdict(set)
        fm_by_sensor: dict[str, set[str]] = defaultdict(set)
        for fm, sensor in graph.edges:
            sensors_by_fm[fm].add(sensor)
            fm_by_sensor[sensor].add(fm)

        for fm in graph.failure_modes:
            pos = sorted(sensors_by_fm.get(fm, set()))
            neg = sorted(all_sensors - set(pos))
            rng.shuffle(pos)
            for i, correct in enumerate(pos[:positives_per_anchor]):
                distractors = rng.sample(neg, min(options_per_question - 1, len(neg)))
                if distractors:
                    generated.append(make_row(asset, "fm2sensor", "positive", fm, correct, distractors, i))
            if pos and neg:
                for i, correct in enumerate(rng.sample(neg, min(negatives_per_anchor, len(neg)))):
                    distractors = rng.sample(pos, min(options_per_question - 1, len(pos)))
                    generated.append(make_row(asset, "fm2sensor", "negative", fm, correct, distractors, i))

        for sensor in graph.sensors:
            pos = sorted(fm_by_sensor.get(sensor, set()))
            neg = sorted(all_fm - set(pos))
            rng.shuffle(pos)
            for i, correct in enumerate(pos[:positives_per_anchor]):
                distractors = rng.sample(neg, min(options_per_question - 1, len(neg)))
                if distractors:
                    generated.append(make_row(asset, "sensor2fm", "positive", sensor, correct, distractors, i))
            if pos and neg:
                for i, correct in enumerate(rng.sample(neg, min(negatives_per_anchor, len(neg)))):
                    distractors = rng.sample(pos, min(options_per_question - 1, len(pos)))
                    generated.append(make_row(asset, "sensor2fm", "negative", sensor, correct, distractors, i))
    return generated


def permute_options(row: NormalizedMCQ, seed: int) -> NormalizedMCQ:
    rng = random.Random(seed)
    original = list(row.options.items())
    contents = [value for _, value in original]
    rng.shuffle(contents)
    labels = [label for label, _ in original]
    new_options = OrderedDict(zip(labels, contents))
    correct_texts = {row.options[label] for label in row.correct_labels if label in row.options}
    new_correct = [label for label, value in new_options.items() if value in correct_texts]
    clone = NormalizedMCQ(**{**asdict(row), "options": new_options, "correct_labels": new_correct})
    return clone


def make_training_rows(
    single_rows: Sequence[NormalizedMCQ],
    graphs: Mapping[str, AssetGraph],
    permutation_copies: int = 1,
    include_graph_synthetic: bool = True,
    seed: int = 42,
) -> list[NormalizedMCQ]:
    rows: list[NormalizedMCQ] = [row for row in single_rows if row.is_single_answer]
    if include_graph_synthetic:
        rows.extend(synthesize_single_answer_rows_from_graph(graphs, seed=seed))
    augmented = list(rows)
    for copy_idx in range(permutation_copies):
        for row in rows:
            augmented.append(permute_options(row, seed + copy_idx * 1000003 + stable_int(row.id)))
    return augmented


# -----------------------------------------------------------------------------
# Prompt construction and token-span supervision
# -----------------------------------------------------------------------------

# Option-Content Listwise SFT: the assistant target and the closing user hint switch
# between a bare "<answer>C</answer>" ("label") and a label+content block
# ("label_content").  Toggled globally so training and inference stay in lock-step;
# ``set_answer_style`` is called once by train.py / evaluate.py from config.
_ANSWER_STYLE = "label"


def set_answer_style(style: str) -> None:
    global _ANSWER_STYLE
    if style not in ("label", "label_content", "score"):
        raise ValueError(
            f"unknown answer_style: {style!r} (expected 'label', 'label_content' or 'score')"
        )
    _ANSWER_STYLE = style


def get_answer_style() -> str:
    return _ANSWER_STYLE


# Strict prompt (§6-8e): the competition test input carries only passage/question/
# options, so asset/direction/polarity/anchor must never be serialized into the
# prompt. They stay available to the trainer as side information for graph
# construction and losses. Ported from the audited IQ2 implementation.
_STRICT_PROMPT = False


def set_strict_prompt(enabled: bool) -> None:
    global _STRICT_PROMPT
    _STRICT_PROMPT = bool(enabled)


def get_strict_prompt() -> bool:
    return _STRICT_PROMPT


# Strict target (IQ2, §13-5): the same policy applied to the assistant <think>
# target. strict_prompt keeps metadata out of the INPUT; this keeps it out of the
# OUTPUT the model is trained to produce. Separate flag because the two were
# introduced in different runs and old adapters must stay reproducible.
_STRICT_TARGET = False


def set_strict_target(enabled: bool) -> None:
    global _STRICT_TARGET
    _STRICT_TARGET = bool(enabled)


def get_strict_target() -> bool:
    return _STRICT_TARGET


# Answer-only LM loss (§6-6a signal dilution). The MCQA <think> target is a fixed
# template that names no relation ("Compare the wording ... best satisfies the
# question"), yet it carries 41 of the 61 supervised tokens per row — two thirds
# of the LM gradient teaches the template, not the answer. Masking it lands the
# gradient on the answer block instead.
#
# Rows with an AUTHORED rationale are exempt: only the RFD rows (§6-6b) have one,
# and their fact sentence lives inside <think>. Masking it would erase the whole
# relation-fact distillation, the same trap strict_target hit in §13-5a.
_LOSS_ON_ANSWER_ONLY = False


def set_loss_on_answer_only(enabled: bool) -> None:
    global _LOSS_ON_ANSWER_ONLY
    _LOSS_ON_ANSWER_ONLY = bool(enabled)


def get_loss_on_answer_only() -> bool:
    return _LOSS_ON_ANSWER_ONLY


# Think specialization (V3 card A0, CODE_SPEC §23). Run K masked the templated
# MCQA <think> on the theory that it was information-free filler, and val fell
# 94.36 -> 71.90 with the damage concentrated on negation (98.00 -> 47.50). The
# template was not filler: its "including any negation or exclusion wording"
# clause IS the model's negation-handling procedure. So the next move is to make
# that procedure *specific* rather than to delete it.
#
# Hard rule: every quoted string must be a literal substring of what the model
# actually sees at inference (passage + question). Naming asset/direction/
# polarity from metadata is forbidden -- none is supplied at test time, so a
# target that names them teaches the model to invent them (§6-8e). Note this
# code never reads row.polarity: the exclusion wording is detected from the
# question text alone, which is exactly the signal available at test time.
_THINK_SPECIALIZE = False


def set_think_specialize(enabled: bool) -> None:
    global _THINK_SPECIALIZE
    _THINK_SPECIALIZE = bool(enabled)


def get_think_specialize() -> bool:
    return _THINK_SPECIALIZE


# Ordered longest/most-specific first: alternation is leftmost-first, so
# "does not" must precede "not" to win at the position where both could start.
_EXCLUSION_WORDING = re.compile(
    r"\b("
    r"all but one"
    r"|least likely"
    r"|does\s+not|do\s+not|doesn'?t|don'?t"
    r"|cannot|can\s?not"
    r"|no\s+bearing"
    r"|not\b"
    r"|never"
    r"|except(?:ing)?"
    r"|exclud(?:e|es|ed|ing)"
    # The corpus's two commonest exclusion phrasings; both were missed by the
    # first draft of this pattern and account for 390 of its 526 misses.
    r"|non-?relevant"
    r"|non-?essential"
    r"|insignificant"
    r"|inconsequential"
    r"|negligible"
    r"|immaterial"
    r"|unnecessary"
    r"|unrelated"
    r"|irrelevant"
    r"|unaffected"
    r"|uninformative"
    r"|unimportant"
    r"|disregard(?:ed|ing)?"
    r"|least"
    r")\b",
    flags=re.IGNORECASE,
)


def question_exclusion_wording(row: NormalizedMCQ) -> str | None:
    """Literal exclusion wording as it appears in the question, or None.

    Searches the question only. The passage describes the asset and can contain
    incidental negations that do not flip what the question asks for.
    """
    match = _EXCLUSION_WORDING.search(row.question or "")
    return match.group(0) if match else None


def question_anchor_quote(row: NormalizedMCQ) -> str | None:
    """The anchor as literally written in the text the model can see.

    Mirrors the anchor lookup in ``_build_strict_user_content``: question first,
    then passage. Returns the matched slice rather than ``row.anchor`` so the
    target quotes the visible spelling, never the canonical metadata one.
    """
    if not row.anchor:
        return None
    for region in (row.question or "", row.passage or ""):
        region = region.strip()
        if not region:
            continue
        found = _case_insensitive_span(region, row.anchor)
        if found is not None:
            return region[found[0] : found[1]]
    return None


def _case_insensitive_span(text: str, needle: str) -> tuple[int, int] | None:
    """Return the first case-insensitive literal occurrence of needle."""
    needle = str(needle or "").strip()
    if not needle:
        return None
    match = re.search(re.escape(needle), text, flags=re.IGNORECASE)
    if match:
        return (match.start(), match.end())

    # Canonical metadata can normalize whitespace. Retry with a flexible
    # whitespace pattern while preserving a literal match for all other chars.
    pieces = [re.escape(piece) for piece in re.split(r"\s+", needle) if piece]
    if not pieces:
        return None
    match = re.search(r"\s+".join(pieces), text, flags=re.IGNORECASE)
    return (match.start(), match.end()) if match else None


USER_INSTRUCTION = """You are solving a physics-grounded industrial FMEA multiple-choice problem.
Use the supplied asset context and knowledge internalized in your model parameters only.
Reason causally in this order: failure mode -> affected physical quantity -> observable sensor response.
Pay special attention to NOT, LEAST, irrelevant, unrelated, and exclusion wording.
Choose exactly one listed option."""

# Strict variant (IQ2): under the strict prompt there IS no supplied asset context,
# so promising one is a leftover that contradicts what the model actually receives.
STRICT_USER_INSTRUCTION = """You are solving a physics-grounded industrial FMEA multiple-choice problem.
Use only the passage, question, and listed options shown below.
Reason about the requested relation between failure modes and observable sensors.
Pay special attention to NOT, LEAST, irrelevant, unrelated, and exclusion wording.
Choose exactly one listed option."""


# General-industrial variant. The FMEA instruction tells the model to reason
# "failure mode -> physical quantity -> sensor response", but the competition's
# 290 `multi_choice_question_answering` rows have no such relation to reason
# about -- they ask about NPSH, grounding, lockout, steam traps. Sending the
# relational instruction with them is an instruction/task mismatch on every
# one of those rows, at training time and at inference time alike.
GENERAL_USER_INSTRUCTION = """You are solving an industrial engineering multiple-choice problem.
Use only the question and the listed options shown below, together with knowledge internalized in your model parameters.
Decide which option the question's own wording most directly supports; several options may be related to the topic, but only one answers the question as asked.
Pay special attention to NOT, LEAST, EXCEPT, and other exclusion wording, and to hedges such as "can", "may" and "typically".
Choose exactly one listed option."""


def is_general_industrial(row: NormalizedMCQ) -> bool:
    """True for rows that are NOT failure-mode/sensor relational questions.

    Two independent markers so the split works on both sides of the pipeline:
    our generated rows carry the metadata flag, and the competition test file
    carries `question_type` (2,758 `open_ended_multi_choice` vs 290
    `multi_choice_question_answering`) -- the router needs no gating layer and
    no heuristic, the label is in the data.
    """
    if (row.metadata or {}).get("industrial_mcqa"):
        return True
    return str(row.question_type).strip() == "multi_choice_question_answering"


def _user_instruction(row: NormalizedMCQ | None = None) -> str:
    if row is not None and is_general_industrial(row):
        return GENERAL_USER_INSTRUCTION
    return STRICT_USER_INSTRUCTION if _STRICT_PROMPT else USER_INSTRUCTION


def _specialized_strict_reasoning(
    row: NormalizedMCQ, answer_label: str, answer_text: str
) -> str | None:
    """Question-grounded variant of the strict <think> target, or None to fall back.

    Returns None when neither the anchor nor exclusion wording can be quoted from
    visible text, so no row ever loses its rationale — the same defensive shape
    the answer-only masking uses. The polarity claim is sound under the §14-2
    distractor policy: a positive question draws its gold from edges and its
    distractors from observed non-edges, and a negative question does the
    reverse, so the gold is on exactly one side of "holds".
    """
    anchor = question_anchor_quote(row)
    exclusion = question_exclusion_wording(row)
    if anchor is None and exclusion is None:
        return None
    if exclusion is None and row.polarity == "negative":
        # The question excludes but this pattern cannot see how it says so. Emitting
        # "no exclusion wording is present" here would train the exact inversion the
        # target is meant to prevent, so hand the row back to the generic template.
        # polarity is used ONLY as this veto -- it never becomes emitted language,
        # so §6-8e still holds: nothing in the output names train-only metadata.
        return None

    parts: list[str] = []
    if anchor is not None:
        parts.append(f'The question is anchored on "{anchor}".')
    if exclusion is not None:
        parts.append(
            f'The wording "{exclusion}" makes this an exclusion question, so the correct '
            "option is the one that does NOT hold."
        )
        verdict = "is the listed option that does not hold"
    else:
        parts.append(
            "No exclusion wording is present, so the correct option is the one that does hold."
        )
        verdict = "is the listed option that holds"
    parts.append(
        f'Checking every listed option, option {answer_label}, "{answer_text}", {verdict}.'
    )
    parts.append(_conclusion(row, answer_label).strip())
    return " ".join(parts)



def _conclusion(row: NormalizedMCQ, answer_label: str) -> str:
    """How the rationale ends.

    Under `answer_style: score` it ends WITHOUT naming the answer. Stating it
    made the letter objective degenerate: the target think is teacher-forced, so
    the answer sat in the model's own context and the K-way cross-entropy fell to
    0.012 within 15 steps -- ln(5) is 1.61, so it was learning nothing and the
    letter slot was merely copying what the reasoning had already written. At
    inference the think is GENERATED, so a wrong reasoning would be copied just
    as faithfully.

    Dropping the conclusion keeps the part run K proved matters (§23 F8: the
    comparison-and-exclusion procedure) and forces the decision into the letter
    logits, where training and inference face the same question.
    """
    if _ANSWER_STYLE == "score":
        return " Weigh each listed option against exactly what the question asks."
    return f" Therefore the correct option is {answer_label}."

def relation_reasoning(row: NormalizedMCQ, answer_label: str) -> str:
    """Rationale used as the assistant <think> target.

    Under ``strict_target`` (IQ2 policy) the rationale is built from task-visible
    text only. Asset, direction, polarity, and anchor supervise the structural
    losses, but they must never become train-only *language* the model is taught
    to emit: at test time none of them is supplied, so a target that names them
    trains the model to invent them. This is 6-8e applied to the output side.
    """
    # Authored rationales come first and are NEVER overridden. Only the RFD rows
    # (§6-6b) carry one, and naming the asset there is the mechanism, not a leak:
    # those rows exist precisely to imprint asset-conditioned relation facts into
    # the weights. Blanket-applying strict_target here would silently erase all
    # 2,684 of them (MCQA rows carry no authored reasoning at all).
    if row.reasoning:
        text = row.reasoning.strip()
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
        return text
    if _STRICT_TARGET:
        answer_text = row.options[answer_label]
        if _THINK_SPECIALIZE:
            specialized = _specialized_strict_reasoning(row, answer_label, answer_text)
            if specialized is not None:
                return specialized
        opening = (
            "Compare the wording of the question with every listed option, including "
            "any negation or exclusion wording. "
        )
        if _ANSWER_STYLE == "score":
            # Neither the letter nor the answer text appears: the reasoning
            # states the procedure, the letter logits state the choice.
            return opening.rstrip() + _conclusion(row, answer_label)
        return (
            opening
            + f'Option {answer_label}, "{answer_text}", best satisfies the question.'
            + _conclusion(row, answer_label)
        )
    if row.reasoning:
        text = row.reasoning.strip()
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
        return text
    answer_text = row.options[answer_label]
    if _ANSWER_STYLE == "score":
        # Same rule as the strict branch: the reasoning may set up the relation
        # but must not name the answer, or the letter objective degenerates into
        # copying it.
        if row.direction == "fm2sensor":
            setup = (
                f'The failure mode "{row.anchor}" can alter operating quantities '
                "that must be observed by a suitable sensor."
            )
        elif row.direction == "sensor2fm":
            setup = (
                f'The sensor "{row.anchor}" responds to physical changes produced '
                "by equipment faults."
            )
        else:
            setup = "Consider the industrial context and every listed choice."
        return setup + _conclusion(row, answer_label)
    if row.direction == "fm2sensor":
        relation = (
            f'The failure mode "{row.anchor}" can alter operating quantities that must be observed by a suitable sensor. '
            f'Within the training FMEA relation for the {row.asset}, "{answer_text}" is '
            + ("a related sensor" if row.polarity == "positive" else "the unrelated sensor among these choices")
            + "."
        )
    elif row.direction == "sensor2fm":
        relation = (
            f'The sensor "{row.anchor}" responds to physical changes produced by equipment faults. '
            f'Within the training FMEA relation for the {row.asset}, "{answer_text}" is '
            + ("a related failure mode" if row.polarity == "positive" else "the unrelated failure mode among these choices")
            + "."
        )
    else:
        relation = f'Considering the industrial context and all choices, "{answer_text}" best satisfies the question.'
    return relation + _conclusion(row, answer_label)


def _closing_instruction() -> str:
    if _ANSWER_STYLE == "label_content":
        return (
            "\n\nBegin the response with <think> reasoning, then finish with exactly one "
            "answer block naming both the option letter and its exact text:\n"
            "<answer>\n<label>LETTER</label>\n<content>the chosen option text</content>\n</answer>"
        )
    return (
        "\n\nBegin the response with <think> and finish with exactly one "
        "<answer>OPTION_LETTER</answer> tag."
    )


def _build_strict_user_content(row: NormalizedMCQ) -> tuple[str, dict[str, tuple[int, int]]]:
    """Strict task-visible prompt: passage / question / options only.

    No metadata headers and no visible entity markers are emitted, because the
    competition test input carries none of them (§6-8). asset/direction/polarity/
    anchor remain available to the trainer as side information; the anchor span is
    recovered by locating the anchor string inside the *visible* text (question
    first, then passage) so CSReg structural + listwise supervision is unchanged.
    Ported from the audited IQ2 implementation.
    """
    chunks: list[str] = []
    spans: dict[str, tuple[int, int]] = {}

    def length() -> int:
        return sum(len(x) for x in chunks)

    def append(value: str) -> tuple[int, int]:
        start = length()
        chunks.append(value)
        return start, length()

    append(_user_instruction(row))

    searchable_regions: list[tuple[int, int, str]] = []
    if row.passage.strip():
        append("\n\nPassage:\n")
        start, end = append(row.passage.strip())
        searchable_regions.append((start, end, row.passage.strip()))

    append("\n\nQuestion:\n")
    q_start, q_end = append(row.question.strip())
    # The general option scorer conditions on the QUESTION, not on a G* anchor:
    # a row asking about NPSH or lockout has no anchor entity, so the relational
    # listwise head skips it and it receives LM loss only. CODE_SPEC 40.
    spans["question"] = (q_start, q_end)
    # Prefer the question over the passage when locating the anchor.
    searchable_regions.insert(0, (q_start, q_end, row.question.strip()))

    if row.anchor:
        for region_start, _region_end, region_text in searchable_regions:
            found = _case_insensitive_span(region_text, row.anchor)
            if found is not None:
                spans["anchor"] = (region_start + found[0], region_start + found[1])
                break

    append("\n\nOptions:")
    for label, option_text in row.options.items():
        append(f"\n{label}. ")
        start, end = append(option_text)
        spans[f"option_{label}"] = (start, end)

    append(_closing_instruction())
    return "".join(chunks), spans


def build_marked_user_content(row: NormalizedMCQ) -> tuple[str, dict[str, tuple[int, int]]]:
    """Return prompt content and character spans for anchor/options in that content."""
    if _STRICT_PROMPT:
        return _build_strict_user_content(row)

    chunks: list[str] = []
    spans: dict[str, tuple[int, int]] = {}

    def append(text: str) -> None:
        chunks.append(text)

    append(_user_instruction(row))
    append(f"\n\nAsset: {row.asset}")
    append(f"\nDirection: {row.direction}")
    append(f"\nQuestion polarity: {row.polarity}")
    if row.anchor:
        append("\nAnchor entity: <<ANCHOR>>")
        start = sum(len(x) for x in chunks)
        append(row.anchor)
        end = sum(len(x) for x in chunks)
        spans["anchor"] = (start, end)
        append("<</ANCHOR>>")
    append("\n\nQuestion:\n")
    append(row.prompt_question)
    append("\n\nOptions:")
    for label, option_text in row.options.items():
        append(f"\n{label}. <<OPTION_{label}>>")
        start = sum(len(x) for x in chunks)
        append(option_text)
        end = sum(len(x) for x in chunks)
        spans[f"option_{label}"] = (start, end)
        append(f"<</OPTION_{label}>>")
    append(_closing_instruction())
    return "".join(chunks), spans


def build_assistant_target(row: NormalizedMCQ) -> str:
    if not row.is_single_answer:
        raise ValueError("Assistant target requires exactly one correct option")
    label = row.answer_label
    assert label is not None
    reasoning = relation_reasoning(row, label)
    if _ANSWER_STYLE == "score":
        # The model reasons, opens the answer block, and stops. The decision is
        # the distribution over the CANDIDATE LETTER TOKENS at the very next
        # position -- read from the LM head itself, with no extra parameters.
        # Training puts a per-option BCE on exactly those logits, so the
        # quantity optimised and the quantity used at inference are the same
        # tensor slot. Probability mass cannot leak to the other ~150k tokens
        # and an option that does not exist cannot be produced.
        return f"<think>\n{reasoning}\n</think>\n<answer>"
    if _ANSWER_STYLE == "label_content":
        # Supervise the option *content*, not only the letter, so the model must
        # ground its choice in the option text (Option-Content Listwise SFT).
        content_text = row.options[label]
        answer_block = f"<answer>\n<label>{label}</label>\n<content>{content_text}</content>\n</answer>"
    else:
        answer_block = f"<answer>{label}</answer>"
    return f"<think>\n{reasoning}\n</think>\n{answer_block}"


def char_span_to_token_span(offsets: Sequence[Sequence[int]], char_span: tuple[int, int]) -> tuple[int, int]:
    char_start, char_end = char_span
    token_indices = [
        i
        for i, (start, end) in enumerate(offsets)
        if end > start and start < char_end and end > char_start
    ]
    if not token_indices:
        return (-1, -1)
    return token_indices[0], token_indices[-1] + 1



def _row_source(row: NormalizedMCQ) -> str:
    """Which generator produced this row -- used only for the coverage report."""
    meta = row.metadata or {}
    for key in (
        "industrial_mcqa",
        "relation_fact",
        "composite_anchor",
        "component_row",
        "pattern_matrix",
        "surface_variant",
        "option_augmented",
    ):
        if meta.get(key):
            return key
    return "corpus"

def tokenize_training_row(
    row: NormalizedMCQ,
    tokenizer: Any,
    graphs: Mapping[str, AssetGraph],
    max_length: int = 1024,
) -> dict[str, Any]:
    content, content_spans = build_marked_user_content(row)
    prompt_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    target = build_assistant_target(row)
    eos = tokenizer.eos_token or ""
    full_text = prompt_chat + target + eos

    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids = tokenizer(
        prompt_chat,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )["input_ids"]
    # Where the answer letter would be predicted FROM: the last token of the
    # target text, before the EOS the template appends. Taking len-1 instead
    # lands on the EOS, whose logits describe what follows the turn, not the
    # answer -- silently training and reading the wrong slot.
    answer_from_index = (
        len(
            tokenizer(
                prompt_chat + target,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )["input_ids"]
        )
        - 1
    )
    labels = list(encoded["input_ids"])
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    content_start = prompt_chat.find(content)
    if content_start < 0:
        raise RuntimeError("Chat template did not preserve user content")
    offsets = encoded.pop("offset_mapping")
    if _LOSS_ON_ANSWER_ONLY and not row.reasoning:
        # Templated <think> carries no relational content: drop it from the LM
        # loss so the gradient lands on the answer block. Authored rationales
        # (RFD) keep theirs. Falls back to the un-masked labels if the answer
        # block cannot be located, so a tokenizer change can never silently
        # leave a row with zero supervised tokens.
        answer_offset = target.find("<answer>")
        if answer_offset >= 0:
            answer_char = len(prompt_chat) + answer_offset
            answer_token = next(
                (i for i, (start, end) in enumerate(offsets) if end > answer_char), None
            )
            if answer_token is not None and answer_token > prompt_len:
                labels[prompt_len:answer_token] = [-100] * (answer_token - prompt_len)
    question_char = content_spans.get("question")
    question_span = (
        char_span_to_token_span(offsets, (content_start + question_char[0], content_start + question_char[1]))
        if question_char
        else (-1, -1)
    )
    anchor_char = content_spans.get("anchor")
    anchor_span = (
        char_span_to_token_span(offsets, (content_start + anchor_char[0], content_start + anchor_char[1]))
        if anchor_char
        else (-1, -1)
    )
    option_spans: list[tuple[int, int]] = []
    for label in row.options:
        char_span = content_spans[f"option_{label}"]
        option_spans.append(
            char_span_to_token_span(offsets, (content_start + char_span[0], content_start + char_span[1]))
        )

    edge_targets = attach_edge_targets(row, graphs)
    option_labels = list(row.options)
    answer_index = option_labels.index(row.answer_label) if row.answer_label in row.options else -1
    option_mask = [int(start >= 0 and end > start) for start, end in option_spans]
    nota_idx = nota_option_index(row)
    # The general head runs on the pre-disabling values: composite and component
    # rows are switched off for the ANCHOR head because their gold is not defined
    # by a single anchor, but they are ordinary "one correct entity" items for a
    # question-conditioned scorer. Relation-fact rows stay off in both heads --
    # their options are literally "Yes"/"No".
    general_option_mask = list(option_mask)
    general_answer_index = answer_index
    if row.metadata.get("relation_fact"):
        general_option_mask = [0] * len(option_spans)
        general_answer_index = -1
    if (
        row.metadata.get("relation_fact")
        or row.metadata.get("composite_anchor")
        or row.metadata.get("component_row")
    ):
        # Yes/No options are not entities: disable option-content supervision so
        # these rows contribute LM loss only (dense fact distillation). Composite
        # rows are excluded for a different reason -- their options are scored
        # against ONE anchor by the listwise and structural heads, but a
        # composite gold is defined by two, so an option can be a true edge of
        # the stored anchor and still be the wrong answer.
        option_mask = [0] * len(option_spans)
        edge_targets = [-1] * len(option_spans)
        answer_index = -1
        nota_idx = -1
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "anchor_span": list(anchor_span),
        "question_span": list(question_span),
        "row_source": _row_source(row),
        "answer_from_index": answer_from_index,
        "letter_token_ids": (
            [
                tokenizer(label, add_special_tokens=False)["input_ids"][0]
                for label in row.options
            ]
            if _ANSWER_STYLE == "score"
            else []
        ),
        "score_gold_index": (
            list(row.options).index(row.answer_label)
            if _ANSWER_STYLE == "score" and row.answer_label in row.options
            else -1
        ),
        "general_option_mask": general_option_mask,
        "general_answer_index": general_answer_index,
        "option_spans": [list(x) for x in option_spans],
        "option_mask": option_mask,
        "edge_targets": edge_targets,
        "direction_id": 0 if row.direction == "fm2sensor" else (1 if row.direction == "sensor2fm" else -1),
        "asset_id": stable_int(row.asset, modulo=2**30),
        # Option-Content Listwise SFT supervision: gold option position and the
        # polarity sign (positive question -> high score is correct; negative/NOT
        # question -> low score is correct).
        "answer_index": answer_index,
        "polarity_id": 1 if row.polarity == "negative" else 0,
        # Abstention: index of the NOTA option (or -1). Its listwise logit comes
        # from the learned abstain gate instead of a content-match score.
        "nota_index": nota_idx,
        # Stratum axes for the held-out probe (fsiq_csreg/stratum_probe.py).
        # Carried on the feature so a probe can bucket without re-deriving them
        # from rows that are no longer in scope by the time training runs.
        "n_options": len(row.options),
        "has_nota": nota_idx >= 0,
        "asset": row.asset,
    }


class CSRegDataCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        # Mutated in place by the stratum probe callback between measurements.
        # A plain dict rather than a copy so the weights a probe computes reach
        # the very next batch; empty means every row weighs 1.0.
        self.stratum_weight_table: dict[str, float] = {}

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            max_length = int(math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of)
        pad_id = int(self.tokenizer.pad_token_id)
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        def pad_sequence(values: list[int], pad_value: int) -> list[int]:
            amount = max_length - len(values)
            if padding_side == "left":
                return [pad_value] * amount + list(values)
            return list(values) + [pad_value] * amount

        batch = {
            "input_ids": torch.tensor(
                [pad_sequence(f["input_ids"], pad_id) for f in features], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [pad_sequence(f["attention_mask"], 0) for f in features], dtype=torch.long
            ),
            "labels": torch.tensor(
                [pad_sequence(f["labels"], -100) for f in features], dtype=torch.long
            ),
        }
        # Entity spans are constructed for right padding. Qwen uses right padding during training.
        if padding_side == "left":
            shifts = [max_length - len(f["input_ids"]) for f in features]
            anchor_spans = [
                [span + shift if span >= 0 else span for span in f["anchor_span"]]
                for f, shift in zip(features, shifts)
            ]
            question_spans = [
                [span + shift if span >= 0 else span for span in f.get("question_span", (-1, -1))]
                for f, shift in zip(features, shifts)
            ]
            shifted_option_spans = [
                [[x + shift if x >= 0 else x for x in span] for span in f["option_spans"]]
                for f, shift in zip(features, shifts)
            ]
        else:
            anchor_spans = [f["anchor_span"] for f in features]
            question_spans = [list(f.get("question_span", (-1, -1))) for f in features]
            shifted_option_spans = [f["option_spans"] for f in features]

        max_options = max(len(feature["option_spans"]) for feature in features)

        def padded(values: list[Any], pad_value: Any) -> list[Any]:
            return values + [pad_value] * (max_options - len(values))

        batch["anchor_span"] = torch.tensor(anchor_spans, dtype=torch.long)
        batch["question_span"] = torch.tensor(question_spans, dtype=torch.long)
        batch["option_spans"] = torch.tensor(
            [padded(spans, [-1, -1]) for spans in shifted_option_spans], dtype=torch.long
        )
        batch["option_mask"] = torch.tensor(
            [padded(f["option_mask"], 0) for f in features], dtype=torch.bool
        )
        batch["general_option_mask"] = torch.tensor(
            [padded(f.get("general_option_mask", f["option_mask"]), 0) for f in features],
            dtype=torch.bool,
        )
        # Answer as a distribution over the CANDIDATE LETTERS at one position.
        # The target ends at "<answer>", so the slot that would emit the letter
        # is the last position of the sequence; its logits, restricted to this
        # row's K letter tokens, ARE the per-option probabilities. One ordinary
        # forward, no extra parameters, and the same tensor slot is read at
        # inference -- there is nothing trained here that inference cannot use.
        if any(f.get("letter_token_ids") for f in features):
            n_letters = max(len(f["letter_token_ids"]) for f in features)
            letters = torch.zeros((len(features), n_letters), dtype=torch.long)
            letter_mask = torch.zeros((len(features), n_letters), dtype=torch.bool)
            answer_slot = torch.zeros((len(features),), dtype=torch.long)
            for i, f in enumerate(features):
                ids = f["letter_token_ids"]
                if ids:
                    letters[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                    letter_mask[i, : len(ids)] = True
                length = len(f["input_ids"])
                offset = (max_length - length) if padding_side == "left" else 0
                index = min(int(f.get("answer_from_index", length - 1)), length - 1)
                answer_slot[i] = offset + index
            batch["letter_token_ids"] = letters
            batch["letter_mask"] = letter_mask
            batch["answer_slot"] = answer_slot
            batch["letter_gold_index"] = torch.tensor(
                [int(f.get("score_gold_index", -1)) for f in features], dtype=torch.long
            )
            if self.stratum_weight_table:
                from .stratum_probe import stratum_of

                batch["loss_weight"] = torch.tensor(
                    [
                        float(self.stratum_weight_table.get(stratum_of(f), 1.0))
                        for f in features
                    ],
                    dtype=torch.float32,
                )
        batch["general_answer_index"] = torch.tensor(
            [int(f.get("general_answer_index", f.get("answer_index", -1))) for f in features],
            dtype=torch.long,
        )
        batch["edge_targets"] = torch.tensor(
            [padded(f["edge_targets"], -1) for f in features], dtype=torch.float32
        )
        batch["direction_id"] = torch.tensor([f["direction_id"] for f in features], dtype=torch.long)
        batch["asset_id"] = torch.tensor([f["asset_id"] for f in features], dtype=torch.long)
        batch["answer_index"] = torch.tensor(
            [int(f.get("answer_index", -1)) for f in features], dtype=torch.long
        )
        batch["polarity_id"] = torch.tensor(
            [int(f.get("polarity_id", 0)) for f in features], dtype=torch.long
        )
        batch["nota_index"] = torch.tensor(
            [int(f.get("nota_index", -1)) for f in features], dtype=torch.long
        )
        # Matrix-regression CSReg (§12) routes graph-row sampling to the assets that
        # actually appear in this micro-batch.  The field is attached by train.py
        # after tokenization, so it is absent for every legacy feature set.
        if any("operator_asset_index" in f for f in features):
            batch["operator_asset_index"] = torch.tensor(
                [int(f.get("operator_asset_index", -1)) for f in features], dtype=torch.long
            )
        return batch


# -----------------------------------------------------------------------------
# Structural projection and Trainer
# -----------------------------------------------------------------------------

def mean_pool_spans(hidden: Any, spans: Any) -> tuple[Any, Any]:
    """Mean-pool [batch, tokens, hidden] using [batch, entities, 2] spans."""
    import torch

    batch_size, entity_count, _ = spans.shape
    pooled = hidden.new_zeros((batch_size, entity_count, hidden.shape[-1]))
    valid = torch.zeros((batch_size, entity_count), dtype=torch.bool, device=hidden.device)
    for b in range(batch_size):
        for e in range(entity_count):
            start, end = [int(x) for x in spans[b, e].tolist()]
            if start >= 0 and end > start and end <= hidden.shape[1]:
                pooled[b, e] = hidden[b, start:end].mean(dim=0)
                valid[b, e] = True
    return pooled, valid


def make_residual_projector(hidden_size: int, rank: int = 64, scale: float = 0.1):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualLowRankProjector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down = nn.Linear(hidden_size, rank, bias=False)
            self.up = nn.Linear(rank, hidden_size, bias=False)
            self.scale = float(scale)
            nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.up.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.up(F.gelu(self.down(x)))
            return F.normalize(x + self.scale * residual, p=2, dim=-1)

    return ResidualLowRankProjector()


def attach_csreg_projector(model: Any, rank: int = 64, scale: float = 0.1) -> Any:
    hidden_size = int(model.config.hidden_size)
    projector = make_residual_projector(hidden_size, rank=rank, scale=scale)
    try:
        reference = model.get_input_embeddings().weight
        target_dtype = reference.dtype if getattr(reference.dtype, "is_floating_point", False) else None
        projector = projector.to(device=reference.device, dtype=target_dtype)
    except Exception:
        pass
    model.csreg_projection = projector
    return model


def make_abstain_gate():
    """Predictor-rejector head: maps real-option score statistics -> abstain logit.

    Input features (per item): [max(s_real), mean(s_real), top1 - top2]. A low max
    score and/or small top1-top2 margin (nothing clearly matches the anchor) yields
    a high abstain logit, so the NOTA option wins the listwise softmax.
    """
    import torch.nn as nn

    gate = nn.Linear(3, 1)
    nn.init.zeros_(gate.weight)
    nn.init.constant_(gate.bias, 0.0)
    return gate


def attach_csreg_abstain_gate(model: Any) -> Any:
    gate = make_abstain_gate()
    try:
        reference = model.get_input_embeddings().weight
        target_dtype = reference.dtype if getattr(reference.dtype, "is_floating_point", False) else None
        gate = gate.to(device=reference.device, dtype=target_dtype)
    except Exception:
        pass
    model.csreg_abstain_gate = gate
    return model


def structural_losses(
    hidden: Any,
    anchor_span: Any,
    option_spans: Any,
    option_mask: Any,
    edge_targets: Any,
    direction_id: Any,
    projector: Any,
    nonedge_margin: float = 0.15,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    anchor_pooled, anchor_valid = mean_pool_spans(hidden, anchor_span[:, None, :])
    option_pooled, option_valid = mean_pool_spans(hidden, option_spans)
    anchor = F.normalize(anchor_pooled[:, 0], p=2, dim=-1)
    options = F.normalize(option_pooled, p=2, dim=-1)
    valid_options = option_mask & option_valid & (edge_targets >= 0)
    valid_anchor = anchor_valid[:, 0]

    zero = hidden.sum() * 0.0
    signature_terms: list[torch.Tensor] = []
    reconstruction_terms: list[torch.Tensor] = []
    positive_terms: list[torch.Tensor] = []
    nonedge_terms: list[torch.Tensor] = []

    for b in range(hidden.shape[0]):
        if not bool(valid_anchor[b]) or int(direction_id[b]) not in (0, 1):
            continue
        mask = valid_options[b]
        if not bool(mask.any()):
            continue
        targets = edge_targets[b, mask]
        option_vecs = options[b, mask]

        if int(direction_id[b]) == 0:  # failure anchor -> sensor options
            projected_parent = projector(anchor[b : b + 1])[0]
            projected = projected_parent[None, :].expand_as(option_vecs)
            residual = option_vecs - targets[:, None] * projected
            m = -2.0 * targets * (projected * residual).sum(dim=-1)
            pos = targets > 0.5
            neg = targets < 0.5
            if bool(pos.any()):
                signature_terms.append((m[pos] ** 2).mean())
                reconstruction_terms.append((residual[pos] ** 2).sum(dim=-1).mean())
                positive_terms.append(((1.0 - F.cosine_similarity(projected[pos], option_vecs[pos])) ** 2).mean())
            if bool(neg.any()):
                cosine = F.cosine_similarity(projected[neg], option_vecs[neg])
                nonedge_terms.append((F.relu(cosine - nonedge_margin) ** 2).mean())
        else:  # failure options -> sensor anchor
            projected_parents = projector(option_vecs)
            pos = targets > 0.5
            neg = targets < 0.5
            if bool(pos.any()):
                # Mean aggregation avoids residual magnitude changing with option count.
                predicted_sensor = projected_parents[pos].mean(dim=0)
                residual = anchor[b] - predicted_sensor
                m = -2.0 * (projected_parents[pos] * residual[None, :]).sum(dim=-1)
                signature_terms.append((m ** 2).mean())
                reconstruction_terms.append((residual ** 2).sum())
                positive_terms.append(
                    ((1.0 - F.cosine_similarity(projected_parents[pos], anchor[b][None, :].expand_as(projected_parents[pos]))) ** 2).mean()
                )
            if bool(neg.any()):
                cosine = F.cosine_similarity(
                    projected_parents[neg], anchor[b][None, :].expand_as(projected_parents[neg])
                )
                nonedge_terms.append((F.relu(cosine - nonedge_margin) ** 2).mean())

    def mean_or_zero(values: list[Any]) -> Any:
        return torch.stack(values).mean() if values else zero

    return {
        "signature": mean_or_zero(signature_terms),
        "reconstruction": mean_or_zero(reconstruction_terms),
        "positive_alignment": mean_or_zero(positive_terms),
        "nonedge_margin": mean_or_zero(nonedge_terms),
    }


def listwise_loss(
    hidden: Any,
    anchor_span: Any,
    option_spans: Any,
    option_mask: Any,
    answer_index: Any,
    polarity_id: Any,
    direction_id: Any,
    projector: Any,
    nota_index: Any = None,
    abstain_gate: Any = None,
    temperature: float = 0.1,
) -> Any:
    """Option-Content Listwise ranking loss with learned abstention (NOTA).

    Every option is scored against the anchor through the CSReg projector (the
    same geometry as ``structural_losses``) and a softmax cross-entropy is applied
    over *all* options at once::

        s_i = <projected(anchor), option_i>                      # direction 0
        s_i = <projected(option_i), anchor>                      # direction 1
        positive question :  logit_i =  s_i / T
        negative question :  logit_i = -s_i / T
        L = -log softmax(logits)[gold]

    Abstention (predictor-rejector): when a NOTA option is present, its logit is
    NOT a content score but ``abstain_gate([max, mean, top1-top2] of the real
    options) / T``. If nothing clearly matches the anchor (low max / small
    margin), the gate raises the NOTA logit so "None of the above" wins. gold is
    kept in the original option index space; invalid options are masked to -inf.

    Grounds the choice in option content, treats 4/8-option items identically,
    compares all options at once, handles positive/negative symmetrically, gives
    gradient to hard distractors, and now learns "all mismatch -> abstain".
    """
    import torch
    import torch.nn.functional as F

    anchor_pooled, anchor_valid = mean_pool_spans(hidden, anchor_span[:, None, :])
    option_pooled, option_valid = mean_pool_spans(hidden, option_spans)
    anchor = F.normalize(anchor_pooled[:, 0], p=2, dim=-1)
    options = F.normalize(option_pooled, p=2, dim=-1)
    valid_options = option_mask.bool() & option_valid
    temp = max(float(temperature), 1e-4)
    neg_inf = torch.finfo(options.dtype).min

    zero = hidden.sum() * 0.0
    terms: list[torch.Tensor] = []
    for b in range(hidden.shape[0]):
        if not bool(anchor_valid[b, 0]):
            continue
        mask = valid_options[b]  # [K_all] over full (padded) option axis
        if int(mask.sum().item()) < 2:  # need at least two options to rank
            continue
        gold = int(answer_index[b])
        if gold < 0 or gold >= mask.shape[0] or not bool(mask[gold]):
            continue

        option_vecs = options[b]  # [K_all, H] (invalid slots are zero, masked below)
        if int(direction_id[b]) == 1:  # options are parents (failure modes) -> project options
            scores = (projector(option_vecs) * anchor[b][None, :]).sum(dim=-1)
        elif int(direction_id[b]) == 0:  # anchor is parent -> project anchor
            projected_anchor = projector(anchor[b : b + 1])[0]
            scores = (option_vecs * projected_anchor[None, :]).sum(dim=-1)
        else:  # unknown direction: raw anchor-option content similarity
            scores = (option_vecs * anchor[b][None, :]).sum(dim=-1)

        sign = -1.0 if int(polarity_id[b]) == 1 else 1.0
        logits = (sign * scores) / temp  # [K_all]

        # Learned abstention: replace the NOTA option's content logit with the gate.
        nota = int(nota_index[b]) if nota_index is not None else -1
        if abstain_gate is not None and 0 <= nota < mask.shape[0] and bool(mask[nota]):
            real_mask = mask.clone()
            real_mask[nota] = False
            # Gate features are detached: the abstain gate reads the score
            # distribution but its gradient never reshapes the representation.
            s_real = scores[real_mask].detach()
            if s_real.numel() >= 1:
                ordered = torch.sort(s_real, descending=True).values
                top1 = ordered[0]
                top2 = ordered[1] if ordered.numel() > 1 else ordered[0]
                feats = torch.stack([s_real.max(), s_real.mean(), top1 - top2])
                abstain_logit = abstain_gate(feats).reshape(())
                logits = logits.clone()
                logits[nota] = abstain_logit / temp
                # Improvement A (representation isolation): on NOTA-gold rows, keep
                # the real-option logits as fixed references so cross-entropy trains
                # ONLY the abstain logit and does not drag the relational
                # representation toward indecision. Non-edge separation is still
                # supervised by structural_losses.nonedge_margin.
                if gold == nota:
                    real_positions = real_mask.nonzero(as_tuple=True)[0]
                    logits[real_positions] = logits[real_positions].detach()

        # Mask invalid (padded) options out of the softmax.
        logits = torch.where(mask, logits, torch.full_like(logits, neg_inf))
        terms.append(
            F.cross_entropy(
                logits[None, :],
                torch.tensor([gold], device=hidden.device, dtype=torch.long),
            )
        )

    return torch.stack(terms).mean() if terms else zero


def make_csreg_trainer_class():
    from transformers import Trainer

    class CSRegTrainer(Trainer):
        def __init__(
            self,
            *args: Any,
            lambda_signature: float = 0.02,
            lambda_reconstruction: float = 0.01,
            lambda_positive: float = 0.01,
            lambda_nonedge: float = 0.02,
            nonedge_margin: float = 0.15,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.lambda_signature = lambda_signature
            self.lambda_reconstruction = lambda_reconstruction
            self.lambda_positive = lambda_positive
            self.lambda_nonedge = lambda_nonedge
            self.nonedge_margin = nonedge_margin
            self.latest_loss_components: dict[str, float] = {}

        def compute_loss(
            self,
            model: Any,
            inputs: MutableMapping[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            anchor_span = inputs.pop("anchor_span")
            option_spans = inputs.pop("option_spans")
            option_mask = inputs.pop("option_mask")
            edge_targets = inputs.pop("edge_targets")
            direction_id = inputs.pop("direction_id")
            inputs.pop("asset_id", None)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
            lm_loss = outputs.loss
            hidden = outputs.hidden_states[-1]
            components = structural_losses(
                hidden=hidden,
                anchor_span=anchor_span,
                option_spans=option_spans,
                option_mask=option_mask,
                edge_targets=edge_targets,
                direction_id=direction_id,
                projector=model.csreg_projection,
                nonedge_margin=self.nonedge_margin,
            )
            total = (
                lm_loss
                + self.lambda_signature * components["signature"]
                + self.lambda_reconstruction * components["reconstruction"]
                + self.lambda_positive * components["positive_alignment"]
                + self.lambda_nonedge * components["nonedge_margin"]
            )
            self.latest_loss_components = {
                "lm": float(lm_loss.detach().cpu()),
                **{name: float(value.detach().cpu()) for name, value in components.items()},
                "total": float(total.detach().cpu()),
            }
            return (total, outputs) if return_outputs else total

        def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
            import torch

            super().save_model(output_dir=output_dir, _internal_call=_internal_call)
            target = Path(output_dir or self.args.output_dir)
            target.mkdir(parents=True, exist_ok=True)
            if hasattr(self.model, "csreg_projection"):
                torch.save(self.model.csreg_projection.state_dict(), target / "csreg_projection.pt")
            (target / "csreg_config.json").write_text(
                json.dumps(
                    {
                        "lambda_signature": self.lambda_signature,
                        "lambda_reconstruction": self.lambda_reconstruction,
                        "lambda_positive": self.lambda_positive,
                        "lambda_nonedge": self.lambda_nonedge,
                        "nonedge_margin": self.nonedge_margin,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    return CSRegTrainer


# -----------------------------------------------------------------------------
# Optional Stage-1 diagnostic: frozen embeddings -> B-hat -> edge AUROC
# -----------------------------------------------------------------------------

def collect_frozen_entity_embeddings(
    rows: Sequence[NormalizedMCQ],
    model: Any,
    tokenizer: Any,
    max_length: int = 768,
    max_rows: int | None = None,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Aggregate last-layer entity embeddings from the frozen/base model.

    Returns ``asset -> {failure_modes: {name: vector}, sensors: {name: vector}}``.
    The function uses only question-visible entity spans and never reads test labels.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    try:
        device = model.get_input_embeddings().weight.device
    except Exception:
        device = next(model.parameters()).device
    buckets: dict[str, dict[str, dict[str, list[np.ndarray]]]] = defaultdict(
        lambda: {"failure_modes": defaultdict(list), "sensors": defaultdict(list)}
    )

    selected = list(rows[:max_rows] if max_rows else rows)
    for row in selected:
        if not row.anchor or row.direction not in {"fm2sensor", "sensor2fm"}:
            continue
        content, content_spans = build_marked_user_content(row)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        content_start = prompt.find(content)
        if content_start < 0:
            continue
        with torch.inference_mode():
            outputs = model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = outputs.hidden_states[-1][0]

        def pooled(key: str) -> np.ndarray | None:
            span = content_spans.get(key)
            if span is None:
                return None
            token_span = char_span_to_token_span(
                offsets, (content_start + span[0], content_start + span[1])
            )
            start, end = token_span
            if start < 0 or end <= start or end > hidden.shape[0]:
                return None
            vector = F.normalize(hidden[start:end].mean(dim=0), dim=-1)
            return vector.float().cpu().numpy()

        anchor_vector = pooled("anchor")
        if anchor_vector is None:
            continue
        if row.direction == "fm2sensor":
            buckets[row.asset]["failure_modes"][canonical_entity(row.anchor)].append(anchor_vector)
            option_type = "sensors"
        else:
            buckets[row.asset]["sensors"][canonical_entity(row.anchor)].append(anchor_vector)
            option_type = "failure_modes"
        for label, option_text in row.options.items():
            vector = pooled(f"option_{label}")
            if vector is not None:
                buckets[row.asset][option_type][canonical_entity(option_text)].append(vector)

    aggregated: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for asset, typed in buckets.items():
        aggregated[asset] = {"failure_modes": {}, "sensors": {}}
        for entity_type in ("failure_modes", "sensors"):
            for name, vectors in typed[entity_type].items():
                mean = np.mean(np.stack(vectors), axis=0)
                mean = mean / max(float(np.linalg.norm(mean)), 1e-12)
                aggregated[asset][entity_type][name] = mean.astype(np.float32)
    return aggregated


def fit_stage1_bhat(
    failure_embeddings: Mapping[str, np.ndarray],
    sensor_embeddings: Mapping[str, np.ndarray],
    graph: AssetGraph,
    steps: int = 400,
    learning_rate: float = 0.03,
    l1_weight: float = 0.02,
    rank: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit the report's vector-node SEM on aggregated frozen entity embeddings."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, roc_auc_score

    seed_everything(seed)
    fms = [fm for fm in graph.failure_modes if fm in failure_embeddings]
    sensors = [s for s in graph.sensors if s in sensor_embeddings]
    if not fms or not sensors:
        return {"asset": graph.asset, "status": "insufficient_embeddings"}

    xf = torch.tensor(np.stack([failure_embeddings[x] for x in fms]), dtype=torch.float32)
    xs = torch.tensor(np.stack([sensor_embeddings[x] for x in sensors]), dtype=torch.float32)
    xf = F.normalize(xf, dim=-1)
    xs = F.normalize(xs, dim=-1)
    hidden = xf.shape[-1]

    logits = nn.Parameter(torch.zeros((len(fms), len(sensors))))
    down = nn.Linear(hidden, rank, bias=False)
    up = nn.Linear(rank, hidden, bias=False)
    nn.init.zeros_(up.weight)
    params = [logits, *down.parameters(), *up.parameters()]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)

    for _ in range(steps):
        b = torch.sigmoid(logits)
        wf = F.normalize(xf + 0.1 * up(F.gelu(down(xf))), dim=-1)
        denom = b.sum(dim=0, keepdim=True).T.clamp_min(1e-4)
        pred = (b.T @ wf) / denom
        loss = ((xs - pred) ** 2).mean() + l1_weight * b.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    labels = np.array(
        [int((fm, sensor) in graph.edges) for fm in fms for sensor in sensors], dtype=np.int64
    )
    result: dict[str, Any] = {
        "asset": graph.asset,
        "failure_modes": fms,
        "sensors": sensors,
        "scores": scores.reshape(len(fms), len(sensors)).tolist(),
        "edge_density": float(labels.mean()),
    }
    if len(np.unique(labels)) == 2:
        result["edge_auroc"] = float(roc_auc_score(labels, scores))
        result["edge_auprc"] = float(average_precision_score(labels, scores))
    return result


# -----------------------------------------------------------------------------
# Inference, permutation consistency, and submission generation
# -----------------------------------------------------------------------------

ANSWER_PATTERNS = (
    # Option-Content Listwise SFT emits <answer><label>C</label><content>..</content></answer>.
    re.compile(r"<label>\s*([A-Z])\s*</label>", re.IGNORECASE),
    re.compile(r"<answer>\s*([A-Z])\s*</answer>", re.IGNORECASE),
    re.compile(r"\\boxed\{\s*([A-Z])\s*\}", re.IGNORECASE),
    re.compile(r'"answer"\s*:\s*"([A-Z])"', re.IGNORECASE),
    re.compile(r"\b(?:answer|option|choice)\s*[:=]?\s*([A-Z])\b", re.IGNORECASE),
)


def extract_answer_letter(text: str, valid_labels: Sequence[str]) -> str | None:
    valid = {str(label).upper() for label in valid_labels}
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        for match in reversed(matches):
            candidate = str(match).upper()
            if candidate in valid:
                return candidate
    stripped = text.strip().upper()
    return stripped if stripped in valid else None


def normalize_scenario(scenario: Any, lexicon: EntityLexicon | None = None) -> NormalizedMCQ:
    if isinstance(scenario, NormalizedMCQ):
        row = scenario
    elif isinstance(scenario, AssetOpsScenario):
        row = normalize_record(scenario.to_dict())
    elif isinstance(scenario, Mapping):
        row = normalize_record(scenario)
    else:
        payload = {
            "id": getattr(scenario, "id"),
            "text": getattr(scenario, "text"),
            "metadata": getattr(scenario, "metadata", {}),
        }
        row = normalize_record(payload)
    if lexicon is not None and not row.anchor:
        row.anchor = extract_anchor(row, lexicon)
    return row


def build_inference_prompt(row: NormalizedMCQ, tokenizer: Any) -> str:
    content, _ = build_marked_user_content(row)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _permuted_row_for_inference(row: NormalizedMCQ, permutation: Sequence[int]) -> tuple[NormalizedMCQ, dict[str, str]]:
    labels = list(row.options)
    values = list(row.options.values())
    shuffled_values = [values[i] for i in permutation]
    options = OrderedDict(zip(labels, shuffled_values))
    label_to_original_text = dict(options)
    clone = NormalizedMCQ(**{**asdict(row), "options": options, "correct_labels": []})
    return clone, label_to_original_text


class CSRegPredictor:
    """Graph-free inference wrapper. Only the fine-tuned model is used at test time."""

    def __init__(
        self,
        base_model_path: str,
        adapter_path: str | None = None,
        graph_bundle_path: str | None = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 320,
        tta_permutations: int = 3,
        inference_batch_size: int = 4,
        seed: int = 42,
        device_map: str | Mapping[str, Any] = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.tta_permutations = max(1, int(tta_permutations))
        self.inference_batch_size = max(1, int(inference_batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        if load_in_4bit and torch.cuda.is_available():
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            torch_dtype=None if quantization_config else (torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            device_map=device_map,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.lexicon = EntityLexicon()
        # Strict Track-1 mode does not load G* or an entity lexicon at inference.
        # graph_bundle_path is accepted only for backward compatibility and intentionally ignored.
        _ = graph_bundle_path

    @property
    def device(self) -> Any:
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.model.parameters()).device

    def _generate_batch(self, prompts: Sequence[str]) -> list[str]:
        import torch

        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self.device)
        prompt_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return [
            self.tokenizer.decode(tokens[prompt_width:], skip_special_tokens=True)
            for tokens in output
        ]

    def _generate(self, prompt: str) -> str:
        return self._generate_batch([prompt])[0]

    def _permutations_for_row(self, row: NormalizedMCQ) -> list[list[int]]:
        labels = list(row.options)
        rng = random.Random(self.seed + stable_int(row.id))
        permutations: list[list[int]] = [list(range(len(labels)))]
        seen = {tuple(permutations[0])}
        max_unique = math.factorial(len(labels)) if len(labels) <= 8 else self.tta_permutations
        while len(permutations) < self.tta_permutations and len(seen) < max_unique:
            perm = list(range(len(labels)))
            rng.shuffle(perm)
            if tuple(perm) not in seen:
                seen.add(tuple(perm))
                permutations.append(perm)
        return permutations

    def _score_label_candidates(self, row: NormalizedMCQ) -> str:
        """Graph-free likelihood fallback when generation does not emit a valid tag."""
        import torch
        import torch.nn.functional as F

        content, _ = build_marked_user_content(row)
        content += "\n\nFor verification, output only one answer tag and no reasoning."
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        best_label = next(iter(row.options))
        best_nll = float("inf")
        for label in row.options:
            if _ANSWER_STYLE == "label_content":
                suffix = (
                    f"<answer>\n<label>{label}</label>\n"
                    f"<content>{row.options[label]}</content>\n</answer>"
                ) + (self.tokenizer.eos_token or "")
            else:
                suffix = f"<answer>{label}</answer>" + (self.tokenizer.eos_token or "")
            suffix_ids = self.tokenizer(suffix, add_special_tokens=False)["input_ids"]
            input_ids = torch.tensor([prompt_ids + suffix_ids], device=self.device)
            labels = input_ids.clone()
            labels[:, : len(prompt_ids)] = -100
            with torch.inference_mode():
                output = self.model(input_ids=input_ids, labels=labels, use_cache=False)
            nll = float(output.loss.detach().cpu())
            if nll < best_nll:
                best_nll = nll
                best_label = label
        return best_label

    def score_letter_logits(self, rows: Sequence[NormalizedMCQ]) -> list[dict[str, str]]:
        """Decide by the candidate-letter logits -- what ``score`` training optimises.

        A ``score`` adapter's target is ``<think>..</think>\\n<answer>`` and then
        EOS: it is trained never to emit the letter as text. Generation therefore
        produces nothing ``extract_answer_letter`` can match, and the generation
        path silently answers from a different decision function entirely. Here
        the reasoning is generated, then one forward over prompt+think+``<answer>``
        reads the logits at that slot restricted to this row's letters -- the same
        tensor slot the K-way cross-entropy trains.
        """
        import torch
        import torch.nn.functional as F

        try:
            from tqdm.auto import tqdm
        except Exception:
            def tqdm(x, **_kwargs):
                return x

        letter_id: dict[str, int] = {}
        for label in ALPHABET:
            ids = self.tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                letter_id[label] = ids[0]

        predictions: list[dict[str, str]] = []
        batch_size = self.inference_batch_size
        starts = range(0, len(rows), batch_size)
        for start in tqdm(starts, total=len(starts), desc="Scoring letters"):
            chunk = list(rows[start : start + batch_size])
            prompts = [build_inference_prompt(row, self.tokenizer) for row in chunk]
            generated = self._generate_batch(prompts)

            prefixes = []
            for prompt, text in zip(prompts, generated):
                cut = text.find("</think>")
                think = text[: cut + len("</think>")] if cut >= 0 else text
                prefixes.append(f"{prompt}{think}\n<answer>")

            encoded = self.tokenizer(
                prefixes, return_tensors="pt", padding=True, add_special_tokens=False
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention = encoded["attention_mask"].to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention).logits

            for index, row in enumerate(chunk):
                labels = [label for label in row.options if label in letter_id]
                if len(labels) < 2:
                    predictions.append({"answer": next(iter(row.options))})
                    continue
                # Left padding is the default, so the final column is the last
                # real token for every row; right padding needs the true length.
                if self.tokenizer.padding_side == "left":
                    position = logits.shape[1] - 1
                else:
                    position = int(attention[index].sum()) - 1
                candidate = torch.tensor(
                    [letter_id[label] for label in labels], device=self.device
                )
                scores = logits[index, position].float().index_select(0, candidate)
                probability = F.softmax(scores, dim=-1)
                predictions.append({"answer": labels[int(torch.argmax(probability))]})
        return predictions

    def predict_rows(self, rows: Sequence[NormalizedMCQ]) -> list[dict[str, str]]:
        if _ANSWER_STYLE == "score":
            # Every caller -- evaluate.py, submit.py, submit_routed.py -- routes
            # through here, so the correction belongs here rather than in each.
            return self.score_letter_logits(rows)

        try:
            from tqdm.auto import tqdm
        except Exception:  # tqdm optional; fall back to a no-op wrapper
            def tqdm(x, **_kwargs):
                return x

        normalized_rows = list(rows)
        votes: list[Counter[str]] = [Counter() for _ in normalized_rows]
        permutations = [self._permutations_for_row(row) for row in normalized_rows]
        rounds = max((len(x) for x in permutations), default=0)

        for round_index in range(rounds):
            work: list[tuple[int, NormalizedMCQ, dict[str, str], str]] = []
            for row_index, row in enumerate(normalized_rows):
                if round_index >= len(permutations[row_index]):
                    continue
                perm_row, label_to_content = _permuted_row_for_inference(
                    row, permutations[row_index][round_index]
                )
                prompt = build_inference_prompt(perm_row, self.tokenizer)
                work.append((row_index, perm_row, label_to_content, prompt))

            batch_starts = range(0, len(work), self.inference_batch_size)
            for start_index in tqdm(
                batch_starts,
                total=len(batch_starts),
                desc=f"Generating (round {round_index + 1}/{rounds})",
            ):
                batch = work[start_index : start_index + self.inference_batch_size]
                generated_batch = self._generate_batch([item[3] for item in batch])
                for (row_index, perm_row, label_to_content, _), generated in zip(batch, generated_batch):
                    answer = extract_answer_letter(generated, list(perm_row.options))
                    if answer is not None:
                        votes[row_index][canonical_entity(label_to_content[answer])] += 1

        predictions: list[dict[str, str]] = []
        for row, counter in zip(normalized_rows, votes):
            if counter:
                selected_content, _ = counter.most_common(1)[0]
                selected_label = next(
                    (label for label, content in row.options.items() if canonical_entity(content) == selected_content),
                    None,
                )
                if selected_label is not None:
                    predictions.append({"answer": selected_label})
                    continue
            predictions.append({"answer": self._score_label_candidates(row)})
        return predictions

    def predict_row(self, row: NormalizedMCQ) -> dict[str, str]:
        if not row.anchor:
            row.anchor = extract_anchor(row, self.lexicon)
        return self.predict_rows([row])[0]

    def predict(self, scenario: Any) -> dict[str, str]:
        row = normalize_scenario(scenario, self.lexicon)
        return self.predict_row(row)


def generate_submission(
    predictor: CSRegPredictor,
    scenarios: Sequence[Any],
    output_path: str | Path,
) -> Path:
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_scenarios = [normalize_scenario(scenario, predictor.lexicon) for scenario in scenarios]
    results = predictor.predict_rows(normalized_scenarios)
    rows: list[dict[str, str]] = []
    for normalized, result in zip(normalized_scenarios, results):
        answer = result["answer"]
        if answer not in normalized.options:
            raise ValueError(f"Invalid answer {answer!r} for scenario {normalized.id}")
        rows.append({"id": normalized.id, "answer": answer})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def load_public_scenarios(path: str | Path) -> list[AssetOpsScenario]:
    scenarios: list[AssetOpsScenario] = []
    for index, raw in enumerate(read_json_records(path)):
        scenario_id = raw.get("id", raw.get("scenario_id", index))
        metadata: dict[str, Any] = {}
        if isinstance(raw.get("metadata"), Mapping):
            metadata.update(raw["metadata"])
        metadata.update({k: raw[k] for k in PUBLIC_EXTRA_FIELDS if k in raw})
        text = str(raw.get("text", raw.get("prompt", "")) or "")
        if not text:
            parts = []
            if raw.get("passage"):
                parts.append(str(raw["passage"]))
            if raw.get("question"):
                parts.append(str(raw["question"]))
            options = _structured_options(raw.get("options"), raw.get("option_ids"))
            if options:
                parts.append("Options:\n" + "\n".join(f"{k}. {v}" for k, v in options.items()))
            text = "\n\n".join(parts)
        scenarios.append(AssetOpsScenario(id=str(scenario_id), text=text, metadata=metadata))
    return scenarios


# Lazy official starter-kit compatible function. Configure through environment variables.
_GLOBAL_PREDICTOR: CSRegPredictor | None = None


def predict(scenario: Any) -> dict[str, str]:
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        base_model = os.environ.get("CSREG_BASE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        adapter = os.environ.get("CSREG_ADAPTER") or None
        graph_bundle = None
        tta = int(os.environ.get("CSREG_TTA", "3"))
        batch_size = int(os.environ.get("CSREG_BATCH_SIZE", "4"))
        _GLOBAL_PREDICTOR = CSRegPredictor(
            base_model_path=base_model,
            adapter_path=adapter,
            graph_bundle_path=graph_bundle,
            tta_permutations=tta,
            inference_batch_size=batch_size,
        )
    return _GLOBAL_PREDICTOR.predict(scenario)


__all__ = [
    "AssetGraph",
    "AssetOpsScenario",
    "CSRegDataCollator",
    "CSRegPredictor",
    "EntityLexicon",
    "NormalizedMCQ",
    "assign_anchors",
    "attach_csreg_projector",
    "attach_edge_targets",
    "build_entity_lexicon",
    "build_inference_prompt",
    "build_marked_user_content",
    "build_train_graphs",
    "collect_frozen_entity_embeddings",
    "find_candidate_data_files",
    "extract_answer_letter",
    "fit_stage1_bhat",
    "generate_submission",
    "group_holdout_split",
    "load_graph_bundle",
    "load_hf_failuresensoriq",
    "load_normalized_records",
    "load_public_scenarios",
    "normalize_scenario",
    "make_csreg_trainer_class",
    "make_training_rows",
    "normalize_record",
    "predict",
    "save_graph_bundle",
    "seed_everything",
    "tokenize_training_row",
]


# ---------------------------------------------------------------------------
# Entity surface generalisation (CODE_SPEC 35-3).
#
# The corpus writes a sensor as "vibration". The competition test writes
# "Vibration sensor" 40 times, and TRAIN and VAL contain that surface form ZERO
# times -- so nothing in our pipeline has ever seen it, and val cannot show the
# damage. A private test pool with more general-vocabulary items makes this
# worse, not better.
#
# The transform is deliberately one-directional. Stripping a prefix would MERGE
# entities -- `audit_option_alias.py` found `compresor pressure/ pressure ratio`
# and `pressure/ pressure ratio` share a surface but hold distinct G*
# neighbourhoods (6 vs 8 edges) -- so only suffixes are added, never removed.
# Verified: no `<sensor> <suffix>` collides with an existing G* entity.
#
# Style is chosen PER ROW, not per option. Test writes all four options in one
# style; varying within a row would hand the model "the odd one out is gold".
SURFACE_SUFFIXES: tuple[str, ...] = (
    "sensor",
    "measurement",
    "monitoring",
    "reading",
    "signal",
    "data",
)
# Already a method, a ratio or an analysis -- "dissolved gas analysis sensor"
# is not English.
_SURFACE_SKIP_TAILS = (
    "analysis",
    "ratio",
    "monitoring",
    "test",
    "inspection",
    "imaging",
    "emission",
    "sensor",
)


def _surface_eligible(entity: str) -> bool:
    text = canonical_entity(entity)
    if not text or "/" in text or "(" in text:
        return False
    return not text.endswith(_SURFACE_SKIP_TAILS)


def apply_surface_suffix(entity: str, suffix: str) -> str:
    return f"{entity} {suffix}" if _surface_eligible(entity) else entity


def synthesize_surface_variant_rows(
    rows: Sequence[NormalizedMCQ],
    graphs: Mapping[str, Any],
    *,
    per_row: int = 1,
    max_rows: int | None = None,
    seed: int = 0,
) -> list[NormalizedMCQ]:
    """Additive copies of existing rows with a uniform sensor surface style.

    Originals are never modified: the benchmark phrasing has to stay dominant,
    and §26 already showed that growing the corpus on an exhausted axis buys
    nothing. These rows widen the surface distribution, not the relation set.
    """
    sensors = {
        canonical_entity(name)
        for graph in graphs.values()
        for name in getattr(graph, "sensors", ())
    }
    known = sensors | {
        canonical_entity(name)
        for graph in graphs.values()
        for name in getattr(graph, "failure_modes", ())
    }
    rng = random.Random(seed)
    generated: list[NormalizedMCQ] = []

    for row in rows:
        if max_rows is not None and len(generated) >= max_rows:
            break
        if not row.is_labeled or row.metadata.get("surface_variant"):
            continue
        # Only rows whose options are sensors can take a sensor-ish suffix.
        values = list(row.options.values())
        targets = [v for v in values if canonical_entity(v) in sensors and _surface_eligible(v)]
        if len(targets) < 2:
            continue
        for _ in range(per_row):
            suffix = rng.choice(SURFACE_SUFFIXES)
            mapped = OrderedDict(
                (label, apply_surface_suffix(value, suffix)
                 if canonical_entity(value) in sensors else value)
                for label, value in row.options.items()
            )
            # A suffix that collapses two options would make the gold ambiguous.
            if len({canonical_entity(v) for v in mapped.values()}) != len(mapped):
                continue
            if any(canonical_entity(v) in known for v in mapped.values() if v not in values):
                continue
            question = row.question
            anchor = row.anchor
            if anchor and canonical_entity(anchor) in sensors and _surface_eligible(anchor):
                suffixed = apply_surface_suffix(anchor, suffix)
                if anchor in question:
                    question = question.replace(anchor, suffixed)
                    anchor = suffixed
            generated.append(
                replace(
                    row,
                    id=f"{row.id}::surface{len(generated)}",
                    question=question,
                    options=mapped,
                    anchor=anchor,
                    metadata={**row.metadata, "surface_variant": suffix},
                )
            )
    return generated


# ---------------------------------------------------------------------------
# General option scoring (CODE_SPEC 40).
#
# `listwise_loss` scores every option against the ANCHOR entity, which is the
# right geometry for "which sensor detects rotor bar damage" and no geometry at
# all for "which term is described as net positive suction head". Rows without
# an anchor are skipped, and that turned out to be 48.1% of the corpus:
#
#     composite_anchor  5,304   gold defined by TWO anchors, so the single-anchor
#                               head is switched off on purpose
#     industrial_mcqa   1,096   general-knowledge rows have no anchor at all
#     relation_fact     2,684   Yes/No options -- content scoring is meaningless,
#                               these stay LM-only by design
#
# 6,400 of those rows are ordinary multiple-choice items that simply need a
# scorer conditioned on the QUESTION instead of on an anchor. Separate
# projections, never the failure-sensor projector: mixing the two would put
# general-industrial options into the relational geometry that run V's
# cross-asset result depends on.
def make_general_option_scorer(hidden_size: int, rank: int = 256):
    import torch
    import torch.nn as nn

    class GeneralOptionScorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj_q = nn.Linear(hidden_size, rank, bias=False)
            self.proj_o = nn.Linear(hidden_size, rank, bias=False)
            nn.init.normal_(self.proj_q.weight, std=0.02)
            nn.init.normal_(self.proj_o.weight, std=0.02)

        def forward(self, question: Any, options: Any) -> Any:
            import torch.nn.functional as F

            q = F.normalize(self.proj_q(question), p=2, dim=-1)
            o = F.normalize(self.proj_o(options), p=2, dim=-1)
            return (q.unsqueeze(1) * o).sum(-1)

    return GeneralOptionScorer()


def attach_general_option_scorer(model: Any, hidden_size: int | None = None, rank: int = 256):
    import torch

    if hidden_size is None:
        hidden_size = int(model.config.hidden_size)
    scorer = make_general_option_scorer(hidden_size, rank=rank)
    device = next(model.parameters()).device
    scorer = scorer.to(device=device, dtype=torch.float32)
    for parameter in scorer.parameters():
        parameter.requires_grad_(True)
    model.general_option_scorer = scorer
    return model


def general_listwise_loss(
    hidden: Any,
    question_span: Any,
    option_spans: Any,
    option_mask: Any,
    answer_index: Any,
    polarity_id: Any,
    scorer: Any,
    temperature: float = 0.1,
) -> Any:
    """Softmax cross-entropy over options, conditioned on the question span.

    Deliberately mirrors `listwise_loss` -- same temperature, same polarity flip,
    same "at least two live options" guard -- so the two heads are comparable and
    a row is supervised by exactly one of them. NOTA needs no special case here:
    with no anchor there is nothing to abstain from matching, and the NOTA string
    is just another option whose content the scorer can learn.
    """
    import torch
    import torch.nn.functional as F

    question_pooled, question_valid = mean_pool_spans(hidden, question_span[:, None, :])
    option_pooled, option_valid = mean_pool_spans(hidden, option_spans)
    valid_options = option_mask.bool() & option_valid
    temp = max(float(temperature), 1e-4)

    total = hidden.new_zeros(())
    counted = 0
    for b in range(hidden.shape[0]):
        if not bool(question_valid[b, 0]):
            continue
        mask = valid_options[b]
        if int(mask.sum().item()) < 2:
            continue
        gold = int(answer_index[b])
        if gold < 0 or gold >= mask.shape[0] or not bool(mask[gold]):
            continue
        scores = scorer(
            question_pooled[b, 0].float().unsqueeze(0), option_pooled[b].float().unsqueeze(0)
        )[0]
        logits = scores / temp
        if int(polarity_id[b]) == 1:
            logits = -logits
        logits = torch.where(mask, logits, torch.full_like(logits, torch.finfo(logits.dtype).min))
        total = total + F.cross_entropy(logits.unsqueeze(0), torch.tensor([gold], device=logits.device))
        counted += 1
    return total / counted if counted else hidden.new_zeros(())


# ---------------------------------------------------------------------------
# Option-count variation for general-industrial rows (CODE_SPEC 41).
#
# `augment_option_counts_from_graph` resolves candidates through G*, so it
# produces nothing for a row whose asset is "industrial" -- measured directly:
# run GEN3 logged `variants_per_epoch_raw: [0, 0]`. Every industrial row was
# therefore seen with the SAME option count and the SAME distractors in every
# epoch, while permutation only shuffled their order.
#
# The candidate pool here is the corpus itself. Each row carries its `topic`,
# and the distractors the generator already drew are in the option strings, so
# the union of options within a topic IS the confusable set -- no dependency on
# the atoms file, and the pool stays consistent with whatever generate_mcqa.py
# last produced.
INDUSTRIAL_NOTA = "none of the above"


def _length_ok(candidate: str, gold_length: int, band: float) -> bool:
    """Keep distractor length comparable to the gold's.

    Measured shortcut: in the `boundary` family the gold (an atom's claim) runs
    103 characters against 60 for its distractors, so "pick the longest option"
    scores well without any industrial knowledge. Sampling inside a band around
    the gold length removes that signal for every family at once, which is why
    there is no per-family special case here.
    """
    if gold_length <= 0:
        return True
    ratio = len(candidate) / gold_length
    return band <= ratio <= (1.0 / band if band > 0 else float("inf"))


def augment_option_counts_for_industrial(
    rows: Sequence[NormalizedMCQ],
    target_counts: Sequence[int] = (4, 5, 6, 7, 8),
    copies_per_row: int = 1,
    length_band: float = 0.6,
    max_reuse_fraction: float = 0.25,
    seed: int = 42,
) -> list[NormalizedMCQ]:
    """Resize and re-populate the option set of general-industrial rows.

    Unlike the graph augmenter this one also SHRINKS: a row generated with eight
    options can appear as a four-option item, so option count is not a stable
    property of a question the model can memorise alongside its answer.
    """
    targets = sorted({int(value) for value in target_counts if int(value) >= 2})
    if not targets or copies_per_row < 1:
        return []

    eligible = [
        row
        for row in rows
        if row.is_single_answer and (row.metadata or {}).get("industrial_mcqa")
    ]
    if not eligible:
        return []

    # Pool per topic, plus a global pool for the `negative` family whose gold is
    # deliberately drawn from a DIFFERENT topic.
    by_topic: dict[str, set[str]] = {}
    everything: set[str] = set()
    for row in eligible:
        topic = str((row.metadata or {}).get("topic", ""))
        bucket = by_topic.setdefault(topic, set())
        for value in row.options.values():
            if canonical_entity(value) == INDUSTRIAL_NOTA:
                continue
            bucket.add(value)
            everything.add(value)

    rng = random.Random(seed)
    reuse_cap = max(1, int(max_reuse_fraction * len(eligible) * copies_per_row))
    usage: Counter = Counter()
    generated: list[NormalizedMCQ] = []

    for row in eligible:
        meta = row.metadata or {}
        topic = str(meta.get("topic", ""))
        family = str(meta.get("family", ""))
        gold_label = row.answer_label
        if gold_label is None:
            continue
        gold_value = row.options[gold_label]
        gold_canonical = canonical_entity(gold_value)
        gold_is_nota = gold_canonical == INDUSTRIAL_NOTA
        # `negative` gold belongs to another topic and IS the answer precisely
        # because of that, so it must never be swapped out or re-drawn.
        pool_source = by_topic.get(topic, set())
        if family == "negative":
            pool_source = pool_source | {
                value for value in everything if canonical_entity(value) != gold_canonical
            }

        # "None of the above" is a fixed 17-character string, so banding on ITS
        # length drags in only short candidates and leaves the row's real
        # options looking systematically longer -- a length shortcut in the
        # other direction. Centre on what the row already shows instead.
        if gold_is_nota:
            others = [len(v) for k, v in row.options.items() if k != gold_label]
            gold_length = int(sum(others) / len(others)) if others else 0
        else:
            gold_length = len(gold_value)
        global_pool = list(everything)
        base_pool = [
            value
            for value in pool_source
            if canonical_entity(value) != gold_canonical and usage[canonical_entity(value)] < reuse_cap
        ]
        if not base_pool:
            continue

        for copy_index in range(copies_per_row):
            target = targets[rng.randrange(len(targets))]
            need = target - 1  # the gold occupies one slot
            # Topic first, then the global pool, but ALWAYS inside the length
            # band. Falling back to unbanded candidates is what re-creates the
            # shortcut: a `boundary` gold is a ~103-character claim while its
            # topic pool holds ~20-character concept names, so an unbanded
            # fallback surrounds a long gold with short distractors. Other
            # families' claim sentences live in the global pool and are the
            # right length, so reaching across topics fixes it without a
            # family-specific branch. If the band still cannot be filled the row
            # is skipped -- the graph augmenter's "skip, never pad" policy.
            banded = [value for value in base_pool if _length_ok(value, gold_length, length_band)]
            if len(banded) < need:
                banded = [
                    value
                    for value in global_pool
                    if canonical_entity(value) != gold_canonical
                    and usage[canonical_entity(value)] < reuse_cap
                    and _length_ok(value, gold_length, length_band)
                ]
            candidates = banded
            if len(candidates) < need:
                continue
            # Inside the band, prefer the closest lengths. Banding alone still
            # left `boundary` golds 15 characters longer than their distractors
            # because the band is wide; taking the nearest 3x candidates and
            # sampling among them keeps the bias near zero without collapsing
            # the draw to one deterministic set.
            if gold_length > 0:
                candidates = sorted(candidates, key=lambda v: abs(len(v) - gold_length))
                candidates = candidates[: max(need * 3, need)]
            rng.shuffle(candidates)
            chosen: list[str] = []
            seen = {gold_canonical}
            for value in candidates:
                if len(chosen) >= need:
                    break
                key = canonical_entity(value)
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(value)
            if len(chosen) != need:
                continue

            values = chosen + [gold_value]
            rng.shuffle(values)
            labels = list(ALPHABET)[: len(values)]
            options = OrderedDict(zip(labels, values))
            correct = [label for label, value in options.items() if value == gold_value]
            if len(correct) != 1:
                continue  # a duplicate surface form would make the gold ambiguous
            for value in chosen:
                usage[canonical_entity(value)] += 1
            generated.append(
                replace(
                    row,
                    id=f"{row.id}::opt{target}_{copy_index}",
                    options=options,
                    correct_labels=correct,
                    metadata={**meta, "industrial_option_variant": target},
                )
            )
    return generated


# ---------------------------------------------------------------------------
# Prefix-shared option scoring (CODE_SPEC 42).
#
# Scoring K options by appending each one to the prefix runs the SAME ~250-token
# prefix K times: measured 37.7 s/it, 7h42m for one run. The prefix is identical
# across branches by construction, so recomputing it is pure waste.
#
# Instead pack prefix + every option's content into ONE sequence and hand the
# model a 4D additive mask in which each content block attends to the prefix and
# to itself, never to a sibling block. Position ids restart at the prefix end for
# each block, so every option is scored in exactly the position it would occupy
# if it had been appended alone -- the same numbers, computed once instead of K
# times.
def build_packed_option_batch(
    prefix_ids: Sequence[int],
    option_ids: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Return input_ids / 4D mask / position_ids / per-option score slots."""
    import torch

    prefix = list(prefix_ids)
    blocks = [list(ids) for ids in option_ids]
    input_ids = prefix + [token for block in blocks for token in block]
    total = len(input_ids)

    neg = torch.finfo(torch.float32).min
    mask = torch.full((total, total), neg, dtype=torch.float32)
    for i in range(len(prefix)):
        mask[i, : i + 1] = 0.0
    positions = list(range(len(prefix)))
    starts: list[int] = []
    cursor = len(prefix)
    for block in blocks:
        starts.append(cursor)
        for offset in range(len(block)):
            row = cursor + offset
            mask[row, : len(prefix)] = 0.0
            mask[row, cursor : cursor + offset + 1] = 0.0
            positions.append(len(prefix) + offset)
        cursor += len(block)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask_4d": mask,
        "position_ids": torch.tensor(positions, dtype=torch.long),
        "block_starts": starts,
        "block_lengths": [len(b) for b in blocks],
        "prefix_length": len(prefix),
    }


def packed_option_scores(
    logits: Any,
    input_ids: Any,
    block_starts: Sequence[int],
    block_lengths: Sequence[int],
    prefix_length: int,
) -> Any:
    """Length-normalised log P of each option's content from a packed forward.

    The predictor of a block's FIRST token is the last prefix token, not
    ``start - 1``: in the packed layout that slot holds the previous option's
    last token, whose own attention never saw this block. Getting this wrong
    silently shifts every option but the first -- it showed up as a 4e-2
    disagreement against per-option forwards, not as an error.
    """
    import torch
    import torch.nn.functional as F

    logprob = F.log_softmax(logits.float(), dim=-1)
    scores = []
    for start, length in zip(block_starts, block_lengths):
        if length <= 0:
            scores.append(logprob.new_zeros(()))
            continue
        targets = torch.arange(start, start + length, device=logits.device)
        predictors = torch.cat(
            [
                torch.tensor([prefix_length - 1], device=logits.device),
                targets[:-1],
            ]
        )
        tokens = input_ids[targets]
        picked = logprob[predictors, :].gather(1, tokens[:, None])[:, 0]
        scores.append(picked.mean())
    return torch.stack(scores)
