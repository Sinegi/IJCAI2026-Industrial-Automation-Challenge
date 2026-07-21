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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

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


def synthesize_nota_rows_from_graph(
    graphs: Mapping[str, AssetGraph],
    options_per_question: int = 5,
    per_anchor: int = 1,
    min_distractors: int = 2,
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

        for fm in graph.failure_modes:
            neg = [s for s in all_sensors if s not in sensors_by_fm.get(fm, set())]
            if len(neg) < min_distractors:
                continue
            for i in range(per_anchor):
                k = min(options_per_question - 1, len(neg))
                distractors = rng.sample(neg, k)
                generated.append(make_nota_row(asset, "fm2sensor", fm, distractors, i))

        for sensor in graph.sensors:
            neg = [f for f in all_fm if f not in fm_by_sensor.get(sensor, set())]
            if len(neg) < min_distractors:
                continue
            for i in range(per_anchor):
                k = min(options_per_question - 1, len(neg))
                distractors = rng.sample(neg, k)
                generated.append(make_nota_row(asset, "sensor2fm", sensor, distractors, i))

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
    if style not in ("label", "label_content"):
        raise ValueError(f"unknown answer_style: {style!r} (expected 'label' or 'label_content')")
    _ANSWER_STYLE = style


def get_answer_style() -> str:
    return _ANSWER_STYLE


USER_INSTRUCTION = """You are solving a physics-grounded industrial FMEA multiple-choice problem.
Use the supplied asset context and knowledge internalized in your model parameters only.
Reason causally in this order: failure mode -> affected physical quantity -> observable sensor response.
Pay special attention to NOT, LEAST, irrelevant, unrelated, and exclusion wording.
Choose exactly one listed option."""


def relation_reasoning(row: NormalizedMCQ, answer_label: str) -> str:
    if row.reasoning:
        text = row.reasoning.strip()
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
        return text
    answer_text = row.options[answer_label]
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
    return relation + f" Therefore the correct option is {answer_label}."


def build_marked_user_content(row: NormalizedMCQ) -> tuple[str, dict[str, tuple[int, int]]]:
    """Return prompt content and character spans for anchor/options in that content."""
    chunks: list[str] = []
    spans: dict[str, tuple[int, int]] = {}

    def append(text: str) -> None:
        chunks.append(text)

    append(USER_INSTRUCTION)
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
    if _ANSWER_STYLE == "label_content":
        append(
            "\n\nBegin the response with <think> reasoning, then finish with exactly one "
            "answer block naming both the option letter and its exact text:\n"
            "<answer>\n<label>LETTER</label>\n<content>the chosen option text</content>\n</answer>"
        )
    else:
        append(
            "\n\nBegin the response with <think> and finish with exactly one "
            "<answer>OPTION_LETTER</answer> tag."
        )
    return "".join(chunks), spans


def build_assistant_target(row: NormalizedMCQ) -> str:
    if not row.is_single_answer:
        raise ValueError("Assistant target requires exactly one correct option")
    label = row.answer_label
    assert label is not None
    reasoning = relation_reasoning(row, label)
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
    labels = list(encoded["input_ids"])
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    content_start = prompt_chat.find(content)
    if content_start < 0:
        raise RuntimeError("Chat template did not preserve user content")
    offsets = encoded.pop("offset_mapping")
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
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "anchor_span": list(anchor_span),
        "option_spans": [list(x) for x in option_spans],
        "option_mask": [int(start >= 0 and end > start) for start, end in option_spans],
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
        "nota_index": nota_option_index(row),
    }


class CSRegDataCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

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
            shifted_option_spans = [
                [[x + shift if x >= 0 else x for x in span] for span in f["option_spans"]]
                for f, shift in zip(features, shifts)
            ]
        else:
            anchor_spans = [f["anchor_span"] for f in features]
            shifted_option_spans = [f["option_spans"] for f in features]

        max_options = max(len(feature["option_spans"]) for feature in features)

        def padded(values: list[Any], pad_value: Any) -> list[Any]:
            return values + [pad_value] * (max_options - len(values))

        batch["anchor_span"] = torch.tensor(anchor_spans, dtype=torch.long)
        batch["option_spans"] = torch.tensor(
            [padded(spans, [-1, -1]) for spans in shifted_option_spans], dtype=torch.long
        )
        batch["option_mask"] = torch.tensor(
            [padded(f["option_mask"], 0) for f in features], dtype=torch.bool
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
            s_real = scores[real_mask]
            if s_real.numel() >= 1:
                ordered = torch.sort(s_real, descending=True).values
                top1 = ordered[0]
                top2 = ordered[1] if ordered.numel() > 1 else ordered[0]
                feats = torch.stack([s_real.max(), s_real.mean(), top1 - top2])
                abstain_logit = abstain_gate(feats).reshape(())
                logits = logits.clone()
                logits[nota] = abstain_logit / temp

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

    def predict_rows(self, rows: Sequence[NormalizedMCQ]) -> list[dict[str, str]]:
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
