"""EQ / fsiq_csreg predictor for the official Industrial Automation Challenge starter kit.

Self-contained: this directory holds everything needed to reproduce our
submission -- the inference code (``fsiq_csreg/``), the config
(``configs/eq_nota_gen.yaml``), the trained LoRA adapter (``adapter/``), the
data (``data/``), and the submission we produced (``reference/``). The only
external download is the base model, ``Qwen/Qwen3-8B`` from the HuggingFace hub.

The official files (``run.py``, ``eval_framework.py``, ``dataset_utils.py``) are
the untouched originals; everything specific to our model lives here.

Two details matter:

* ``answer_style`` / ``strict_prompt`` are read back from the adapter's
  ``csreg_config.json``. This adapter was trained with ``answer_style: score``,
  which means it is trained never to write the answer letter as text -- the
  decision is read from the letter logits at the ``<answer>`` slot, the same
  tensor position the training loss optimizes. Decoding it with plain
  generation would silently answer from a different decision function.
* ``CompetitionKit.run_predictions`` calls the predictor one scenario at a time.
  An 8B model answered row-by-row is far slower than the batched path it was
  built for, so the first call warm-starts: the whole dataset is predicted in
  batches of ``inference.batch_size`` and cached by id, and every later call is
  a dict lookup. Set ``EQ_DATASET`` to the same file as the config's
  ``dataset.dataset_path`` to enable it (``run_eq.sh`` does this for you).

Environment variables (all optional -- the defaults resolve inside this directory):

    EQ_CONFIG          inference YAML             (default: configs/eq_nota_gen.yaml)
    EQ_ADAPTER         LoRA adapter dir           (default: adapter/; "none"/"base"
                                                   runs the base model unmodified)
    EQ_DATASET         dataset to warm-start on   (default: none -> row-by-row)
    EQ_TTA             TTA permutations           (default: config inference.tta_permutations)
    EQ_BATCH_SIZE      inference batch size       (default: config inference.batch_size)
    EQ_MAX_NEW_TOKENS  generation budget          (default: config inference.max_new_tokens)
    EQ_LOAD_IN_4BIT    "0" loads bf16 instead of 4-bit NF4 (needs ~18GB VRAM)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

KIT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = "configs/eq_nota_gen.yaml"
DEFAULT_ADAPTER = "adapter"

# The starter kit imports this file by path, so the vendored fsiq_csreg package
# is not necessarily importable yet.
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))


def _resolve(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (KIT_DIR / path)


def _resolve_config_path() -> Path:
    path = _resolve(os.environ.get("EQ_CONFIG") or DEFAULT_CONFIG)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path} (set EQ_CONFIG)")
    return path


def _resolve_adapter(cfg: dict) -> str | None:
    """``EQ_ADAPTER``, else the bundled adapter/, else where training would have written it."""

    raw = os.environ.get("EQ_ADAPTER")
    if raw is not None:
        if raw.strip().lower() in {"none", "base", "null", ""}:
            return None
        path = _resolve(raw)
    else:
        path = _resolve(DEFAULT_ADAPTER)
        if not path.exists():
            path = _resolve(Path(cfg["training"]["output_dir"]) / "final_adapter")
    if not path.exists():
        raise FileNotFoundError(f"Adapter not found: {path} (set EQ_ADAPTER)")
    return str(path)


def _saved_csreg_config(adapter: str | None) -> dict:
    """Settings baked in at training time, written next to the adapter weights."""

    if not adapter:
        return {}
    path = Path(adapter) / "csreg_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def resolve_answer_style(cfg: dict, adapter: str | None) -> str:
    """The adapter's own answer_style wins: it decides how the answer is decoded."""

    saved = _saved_csreg_config(adapter)
    if saved.get("answer_style"):
        return str(saved["answer_style"])
    return str((cfg.get("listwise") or {}).get("answer_style", "label"))


def resolve_strict_prompt(cfg: dict, adapter: str | None) -> bool:
    """Adapters trained before the strict prompt must keep the prompt they saw."""

    saved = _saved_csreg_config(adapter)
    if "strict_prompt" in saved:
        return bool(saved["strict_prompt"])
    return bool((cfg.get("prompt") or {}).get("strict", False))


def _settings() -> tuple[dict, str | None, dict[str, Any]]:
    config_path = _resolve_config_path()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    adapter = _resolve_adapter(cfg)
    inference = cfg.get("inference", {})
    resolved = {
        "config": str(config_path),
        "base_model": cfg["model"]["base_model"],
        "adapter": adapter,
        "answer_style": resolve_answer_style(cfg, adapter),
        "strict_prompt": resolve_strict_prompt(cfg, adapter),
        "seed": int(cfg.get("seed", 42)),
        "tta_permutations": int(os.environ.get("EQ_TTA") or inference["tta_permutations"]),
        "batch_size": int(os.environ.get("EQ_BATCH_SIZE") or inference["batch_size"]),
        "max_new_tokens": int(os.environ.get("EQ_MAX_NEW_TOKENS") or inference["max_new_tokens"]),
        "load_in_4bit": os.environ.get("EQ_LOAD_IN_4BIT", "1") != "0",
    }
    return cfg, adapter, resolved


def describe() -> dict[str, Any]:
    """Resolved settings, without loading the model. Useful as a pre-flight check."""

    _, _, resolved = _settings()
    return {
        "kit_dir": str(KIT_DIR),
        **resolved,
        "warm_start_dataset": os.environ.get("EQ_DATASET") or None,
    }


_PREDICTOR: Any = None
_CACHE: dict[str, str] = {}
_WARMED = False


def get_predictor():
    """Build the CSRegPredictor once."""

    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR

    from fsiq_csreg.core import (
        CSRegPredictor,
        seed_everything,
        set_answer_style,
        set_strict_prompt,
    )

    _, _, resolved = _settings()
    seed_everything(resolved["seed"])
    set_answer_style(resolved["answer_style"])
    set_strict_prompt(resolved["strict_prompt"])

    logger.info(
        "Loading %s + adapter=%s (answer_style=%s strict_prompt=%s)",
        resolved["base_model"], resolved["adapter"],
        resolved["answer_style"], resolved["strict_prompt"],
    )
    _PREDICTOR = CSRegPredictor(
        base_model_path=resolved["base_model"],
        adapter_path=resolved["adapter"],
        load_in_4bit=resolved["load_in_4bit"],
        max_new_tokens=resolved["max_new_tokens"],
        tta_permutations=resolved["tta_permutations"],
        inference_batch_size=resolved["batch_size"],
        seed=resolved["seed"],
    )
    return _PREDICTOR


def predict_batch(scenarios: list[Any]) -> list[dict[str, str]]:
    """Batched prediction. This is the path the model was tuned for."""

    from fsiq_csreg.core import normalize_scenario

    predictor = get_predictor()
    rows = [normalize_scenario(s, predictor.lexicon) for s in scenarios]
    outputs = predictor.predict_rows(rows)
    for row, out in zip(rows, outputs):
        answer = out["answer"]
        if answer not in row.options:
            raise ValueError(f"Invalid answer {answer!r} for scenario {row.id}")
        _CACHE[str(row.id)] = answer
    return [{"answer": out["answer"]} for out in outputs]


def warm_start(dataset_path: str | Path) -> int:
    """Predict a whole dataset in batches and cache it by id."""

    from fsiq_csreg.core import load_public_scenarios

    scenarios = load_public_scenarios(dataset_path)
    logger.info("Warm-starting on %s scenarios from %s", len(scenarios), dataset_path)
    predict_batch(scenarios)
    return len(scenarios)


def _ensure_warm() -> None:
    global _WARMED
    if _WARMED:
        return
    _WARMED = True  # set first: a failed warm-up must not retry on every row
    dataset = os.environ.get("EQ_DATASET")
    if not dataset:
        logger.warning(
            "EQ_DATASET is not set, so every scenario is predicted on its own. "
            "Point it at the same file as dataset.dataset_path for the batched path."
        )
        return
    try:
        warm_start(dataset)
    except Exception:
        logger.exception("Warm-start failed; falling back to per-scenario prediction")


def predict(scenario: Any) -> dict[str, str]:
    """Starter-kit entry point: one scenario in, ``{"answer": "<letter>"}`` out."""

    _ensure_warm()
    key = str(getattr(scenario, "id", ""))
    cached = _CACHE.get(key)
    if cached is not None:
        return {"answer": cached}
    return predict_batch([scenario])[0]


if __name__ == "__main__":
    print(json.dumps(describe(), indent=2, ensure_ascii=False))
