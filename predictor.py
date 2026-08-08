"""Inference for the Industrial Automation Challenge, Track 1.

This module is the single implementation of our decision procedure. It serves
three callers, so there is nothing to keep in sync:

* the organizers' starter kit, through ``predict(scenario)``;
* the command line, ``python predictor.py --data data/test.jsonl``;
* the Kaggle notebook, which inlines the blocks marked ``SHARED:`` below rather
  than importing this file, so that it stays runnable with no repository
  attached. ``tools/build_notebook.py`` performs that splice, so the notebook
  cannot drift from this file.

## How the answer is decided

The supervised target ends at ``<think>...</think>\\n<answer>`` followed by EOS,
so the model is never trained to write the option letter as text and parsing its
output finds nothing. Training puts a K-way cross entropy on the logits at the
single position after ``<answer>``, restricted to the letters that item offers.
Inference reproduces exactly that: generate the reasoning trace, run one forward
pass over ``prompt + trace + "\\n<answer>"``, restrict the final-position logits
to this item's letters, take the argmax. Training and inference read the same
tensor position.

## Environment variables

    EQ_BASE_MODEL     base weights            (default: Qwen/Qwen3-8B)
    EQ_ADAPTER        LoRA adapter directory  (default: ./adapter)
    EQ_DATASET        dataset to warm-start on when driven by the starter kit
    EQ_BATCH_SIZE     inference batch size    (default: from GPU memory)
    EQ_MAX_NEW_TOKENS generation budget       (default: 192, the training value)
    EQ_LOAD_IN_4BIT   "0" loads bf16/fp16 instead of 4-bit NF4 (needs ~18GB)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

REPO_DIR = Path(__file__).resolve().parent

# Batches differ in prompt length, so the allocator sees a different shape most
# steps; expandable segments keep that from fragmenting the pool. Set before
# torch is imported, which is why it sits at module scope.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_ADAPTER = REPO_DIR / "adapter"


# ===== SHARED:prompt =====
# The prompt, reproduced exactly as training built it. Under the strict prompt
# the item shows only what the competition input carries -- instruction,
# passage, question, options -- with no metadata header and no entity markers,
# because the evaluation items have none of those.

STRICT_USER_INSTRUCTION = """You are solving a physics-grounded industrial FMEA multiple-choice problem.
Use only the passage, question, and listed options shown below.
Reason about the requested relation between failure modes and observable sensors.
Pay special attention to NOT, LEAST, irrelevant, unrelated, and exclusion wording.
Choose exactly one listed option."""

GENERAL_USER_INSTRUCTION = """You are solving an industrial engineering multiple-choice problem.
Use only the question and the listed options shown below, together with knowledge internalized in your model parameters.
Decide which option the question's own wording most directly supports; several options may be related to the topic, but only one answers the question as asked.
Pay special attention to NOT, LEAST, EXCEPT, and other exclusion wording, and to hedges such as "can", "may" and "typically".
Choose exactly one listed option."""

# Still asks for an <answer>LETTER</answer> tag, because that is the string
# training used and therefore the string that puts the model in the state
# training measured. The tag is never parsed.
CLOSING_INSTRUCTION = (
    "\n\nBegin the response with <think> and finish with exactly one "
    "<answer>OPTION_LETTER</answer> tag."
)


def is_general_industrial(row: dict) -> bool:
    """The 290 general items, told apart by the field the data already carries."""
    return row["question_type"] == "multi_choice_question_answering"


def build_user_content(row: dict) -> str:
    parts = [GENERAL_USER_INSTRUCTION if is_general_industrial(row) else STRICT_USER_INSTRUCTION]
    if row["passage"]:
        parts.append("\n\nPassage:\n")
        parts.append(row["passage"])
    parts.append("\n\nQuestion:\n")
    parts.append(row["question"])
    parts.append("\n\nOptions:")
    for label, text in row["options"].items():
        parts.append(f"\n{label}. ")
        parts.append(text)
    parts.append(CLOSING_INSTRUCTION)
    return "".join(parts)


def build_prompt(row: dict, tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": build_user_content(row)}],
        tokenize=False,
        add_generation_prompt=True,
    )
# ===== /SHARED:prompt =====


# ===== SHARED:data =====
def normalize_row(raw: dict) -> dict:
    """The four fields the prompt is built from, normalised as during training."""
    passage = str(raw.get("passage", "") or "").strip()
    question = str(raw.get("question", "") or "").strip()
    if not question:                       # a passage-only item becomes the question
        question, passage = passage, ""
    options = OrderedDict(
        (str(k).strip().upper(), str(v).strip()) for k, v in raw["options"].items()
    )
    if not options:
        raise ValueError(f"record {raw.get('id')!r} has no options")
    return {
        "id":            str(raw["id"]),
        "question_type": str(raw.get("question_type", "") or "").strip(),
        "passage":       passage,
        "question":      question,
        "options":       options,
    }


def load_rows(path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(normalize_row(json.loads(line)))
    return rows
# ===== /SHARED:data =====


# ===== SHARED:decode =====
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_MAX_NEW_TOKENS = 192          # the training-time budget; traces never need more


def letter_token_ids(tokenizer: Any) -> dict[str, int]:
    """Option letters that occupy a single token, which is all of A-Z here."""
    table = {}
    for letter in ALPHABET:
        ids = tokenizer(letter, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            table[letter] = ids[0]
    return table


def score_chunk(
    chunk: list[dict],
    model: Any,
    tokenizer: Any,
    device: Any,
    letter_id: dict[str, int],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> list[dict]:
    import torch
    import torch.nn.functional as F

    prompts = [build_prompt(row, tokenizer) for row in chunk]

    # (1) generate the reasoning trace
    encoded = tokenizer(prompts, return_tensors="pt", add_special_tokens=False,
                        padding=True).to(device)
    prompt_width = encoded["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    texts = [tokenizer.decode(t[prompt_width:], skip_special_tokens=True) for t in generated]

    thinks, prefixes = [], []
    for prompt, text in zip(prompts, texts):
        cut = text.find("</think>")
        think = text[: cut + len("</think>")] if cut >= 0 else text
        thinks.append(think)
        prefixes.append(f"{prompt}{think}\n<answer>")

    # (2) one forward pass; read the letter logits at the <answer> slot.
    #
    # Only the final position is used, so the lm_head is asked for that one row
    # of the vocabulary. The full [batch, length, 151936] tensor is ~4.9 GB at
    # batch 16 and length 1000 -- larger than the quantised model -- and every
    # position but the last is computed and thrown away. Left padding puts the
    # last real token in the last column for every row in the batch.
    scored = tokenizer(prefixes, return_tensors="pt", add_special_tokens=False,
                       padding=True).to(device)
    with torch.no_grad():
        try:
            logits = model(input_ids=scored["input_ids"],
                           attention_mask=scored["attention_mask"],
                           logits_to_keep=1).logits
        except TypeError:                              # older transformers spelling
            try:
                logits = model(input_ids=scored["input_ids"],
                               attention_mask=scored["attention_mask"],
                               num_logits_to_keep=1).logits
            except TypeError:
                logits = model(input_ids=scored["input_ids"],
                               attention_mask=scored["attention_mask"]).logits
    last = logits[:, -1, :]        # correct whether 1 or T positions came back

    out = []
    for index, row in enumerate(chunk):
        labels = [label for label in row["options"] if label in letter_id]
        candidate = torch.tensor([letter_id[label] for label in labels], device=device)
        probs = F.softmax(last[index].float().index_select(0, candidate), dim=-1)
        best = int(torch.argmax(probs))
        out.append({
            "id":     row["id"],
            "answer": labels[best],
            "margin": float(probs[best] - probs.topk(2).values[1]) if len(labels) > 1 else 1.0,
            "think":  thinks[index],
        })
    return out


def score_chunk_or_split(chunk: list[dict], **kwargs) -> list[dict]:
    """Halve the batch and retry on OOM instead of losing the whole run.

    The batch size that fits the first items is not guaranteed to fit a later
    batch of long passages with eight options each. Over a multi-hour run that
    is a real risk, and the cost of the fallback is a slightly different padding
    width for the affected rows -- the decision itself is unchanged.
    """
    import gc
    import torch

    try:
        return score_chunk(chunk, **kwargs)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        # Not every OOM arrives as OutOfMemoryError; some paths raise a plain
        # RuntimeError. Anything else is a real bug and must not be retried.
        if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error):
            raise
        if len(chunk) == 1:
            raise
        # The traceback keeps the failed frame -- and every activation tensor it
        # references -- alive, so empty_cache() on its own frees almost nothing
        # and the retry OOMs at the smaller size too.
        error.__traceback__ = None
        gc.collect()
        torch.cuda.empty_cache()
        half = len(chunk) // 2
        print(f"OOM on {len(chunk)} rows — retrying as {half} + {len(chunk) - half}")
        return (score_chunk_or_split(chunk[:half], **kwargs)
                + score_chunk_or_split(chunk[half:], **kwargs))


def predict_rows(
    rows: Sequence[dict],
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    progress: Callable[[Iterable], Iterable] = lambda x: x,
) -> list[dict]:
    letter_id = letter_token_ids(tokenizer)
    missing = {label for row in rows for label in row["options"]} - set(letter_id)
    if missing:
        raise ValueError(f"option letters are not single tokens: {sorted(missing)}")

    predictions: list[dict] = []
    for start in progress(range(0, len(rows), batch_size)):
        predictions.extend(score_chunk_or_split(
            list(rows[start : start + batch_size]),
            model=model, tokenizer=tokenizer, device=device,
            letter_id=letter_id, max_new_tokens=max_new_tokens,
        ))
    return predictions
# ===== /SHARED:decode =====


# --------------------------------------------------------------------------- #
# Model loading. Not shared with the notebook: there the paths come from mounted
# Kaggle Inputs, here from this directory and the HuggingFace hub.
# --------------------------------------------------------------------------- #

def load_model(base_model: str | None = None, adapter: str | Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    base_model = base_model or os.environ.get("EQ_BASE_MODEL") or DEFAULT_BASE_MODEL
    adapter = Path(adapter or os.environ.get("EQ_ADAPTER") or DEFAULT_ADAPTER)
    if not (adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            f"No adapter at {adapter}. It is distributed separately because of its "
            "size — see adapter/README.md, or set EQ_ADAPTER."
        )

    # The adapter records the decision style it was trained under. A `label`
    # adapter run through the logit path below would produce a plausible-looking
    # file that is silently wrong, so the mismatch is made fatal here.
    saved = json.loads((adapter / "csreg_config.json").read_text(encoding="utf-8"))
    if saved.get("answer_style") != "score":
        raise ValueError(f"this predictor implements answer_style 'score', not {saved.get('answer_style')!r}")
    if not saved.get("strict_prompt"):
        raise ValueError("this predictor builds the strict prompt only")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pin the prompt format to the template the adapter was trained with, so a
    # differently-versioned copy of the base model cannot reformat the prompt.
    template = adapter / "chat_template.jinja"
    if template.exists():
        saved_template = template.read_text(encoding="utf-8")
        if (tokenizer.chat_template or "").strip() != saved_template.strip():
            logger.warning("base-model chat template differs from the trained one — using the trained one")
            tokenizer.chat_template = saved_template

    quantization_config = None
    if os.environ.get("EQ_LOAD_IN_4BIT", "1") != "0":
        # Not torch.cuda.is_bf16_supported(): since torch 2.9 it counts emulated
        # bf16 and answers True on a T4, which has no bf16 hardware. Ampere
        # (compute capability 8.0) is where real bf16 starts.
        has_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if has_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        dtype=None if quantization_config else torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    device = model.get_input_embeddings().weight.device
    logger.info("loaded %s + %s on %s", base_model, adapter, device)
    return model, tokenizer, device


def default_batch_size() -> int:
    import torch

    override = os.environ.get("EQ_BATCH_SIZE")
    if override:
        return int(override)
    if not torch.cuda.is_available():
        return 4
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return 64 if gpu_gb >= 24 else 16


# --------------------------------------------------------------------------- #
# Starter-kit interface: eval_framework calls predict(scenario) one item at a
# time. An 8B model answered row by row is far slower than the batched path it
# was built for, so the first call predicts the whole dataset in batches and
# caches it by id; every later call is a dict lookup. Point EQ_DATASET at the
# same file the kit is evaluating to enable it (run.sh does this).
# --------------------------------------------------------------------------- #

_STATE: dict[str, Any] = {"model": None, "tokenizer": None, "device": None}
_CACHE: dict[str, str] = {}
_WARMED = False


def _ensure_model():
    if _STATE["model"] is None:
        _STATE["model"], _STATE["tokenizer"], _STATE["device"] = load_model()
    return _STATE["model"], _STATE["tokenizer"], _STATE["device"]


def predict_batch(rows: Sequence[dict]) -> list[dict]:
    model, tokenizer, device = _ensure_model()
    results = predict_rows(rows, model, tokenizer, device, default_batch_size())
    for result in results:
        _CACHE[result["id"]] = result["answer"]
    return results


def _scenario_to_row(scenario: Any) -> dict:
    raw = scenario if isinstance(scenario, dict) else {
        "id": getattr(scenario, "id", ""),
        **(getattr(scenario, "metadata", None) or {}),
    }
    return normalize_row(raw)


def _ensure_warm() -> None:
    global _WARMED
    if _WARMED:
        return
    _WARMED = True          # set first: a failed warm-up must not retry every row
    dataset = os.environ.get("EQ_DATASET")
    if not dataset:
        logger.warning(
            "EQ_DATASET is not set, so every scenario is predicted on its own. "
            "Point it at the dataset being evaluated for the batched path."
        )
        return
    try:
        predict_batch(load_rows(dataset))
    except Exception:
        logger.exception("warm-start failed; falling back to per-scenario prediction")


def predict(scenario: Any) -> dict[str, str]:
    """Starter-kit entry point: one scenario in, ``{"answer": "<letter>"}`` out."""
    _ensure_warm()
    cached = _CACHE.get(str(getattr(scenario, "id", "")))
    if cached is not None:
        return {"answer": cached}
    return {"answer": predict_batch([_scenario_to_row(scenario)])[0]["answer"]}


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def write_submission(rows: Sequence[dict], results: Sequence[dict], output: Path) -> None:
    by_id = {row["id"]: row for row in rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for result in results:
            # An answer outside the item's own options aborts the write. Falling
            # back to the first option would turn a broken run into a
            # plausible-looking file.
            if result["answer"] not in by_id[result["id"]]["options"]:
                raise ValueError(f"{result['answer']!r} is not an option of item {result['id']}")
            writer.writerow({"id": result["id"], "answer": result["answer"]})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=str(REPO_DIR / "data/test.jsonl"))
    parser.add_argument("--output", default=str(REPO_DIR / "results/submission.csv"))
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N items only (smoke test)")
    parser.add_argument("--describe", action="store_true", help="resolve paths and exit, no model load")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    adapter = Path(args.adapter or os.environ.get("EQ_ADAPTER") or DEFAULT_ADAPTER)
    if args.describe:
        print(json.dumps({
            "base_model": args.base_model or os.environ.get("EQ_BASE_MODEL") or DEFAULT_BASE_MODEL,
            "adapter": str(adapter),
            "adapter_present": (adapter / "adapter_model.safetensors").exists(),
            "data": args.data,
            "output": args.output,
        }, indent=2))
        return 0

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} items from {args.data}")

    model, tokenizer, device = load_model(args.base_model, adapter)
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(x, **_):
            return x

    results = predict_rows(
        rows, model, tokenizer, device,
        args.batch_size or default_batch_size(),
        progress=lambda it: tqdm(it, desc="scoring"),
    )

    output = Path(args.output)
    write_submission(rows, results, output)
    margins = sorted(r["margin"] for r in results)
    print(f"{output} — {len(results)} rows")
    print(f"decision margin: median {margins[len(margins) // 2]:.3f}, min {margins[0]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
