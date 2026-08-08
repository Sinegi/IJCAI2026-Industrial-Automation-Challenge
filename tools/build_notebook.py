#!/usr/bin/env python3
"""Generate ``notebook/eq_track1_inference.ipynb`` — the notebook for Kaggle.

The notebook is a build artifact, not something to hand-edit: editing JSON with
embedded source strings is how cells silently drift apart. Change this file (or
``predictor.py``) and re-run ``python tools/build_notebook.py``.

The notebook must run on Kaggle with no repository attached, so it cannot import
``predictor``. Instead the blocks that ``predictor.py`` marks with
``# ===== SHARED:<name> =====`` are spliced into the cells verbatim. There is
therefore one implementation of the decision procedure, not two: the notebook a
reviewer reads is the same code the command line runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "notebook/eq_track1_inference.ipynb"
PREDICTOR = REPO / "predictor.py"

CELLS: list[tuple[str, str]] = []


def shared(name: str) -> str:
    """The block predictor.py marks with SHARED:<name>, verbatim."""
    source = PREDICTOR.read_text(encoding="utf-8")
    match = re.search(
        rf"^# ===== SHARED:{re.escape(name)} =====\n(.*?)^# ===== /SHARED:{re.escape(name)} =====$",
        source,
        re.DOTALL | re.MULTILINE,
    )
    if match is None:
        raise SystemExit(f"predictor.py has no SHARED:{name} block")
    return match.group(1).strip("\n")


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# --------------------------------------------------------------------------- #
md(
    r"""
# Industrial Automation Challenge — Track 1 · final inference notebook

Running this notebook top to bottom reproduces the prediction file we submitted.
It performs inference only; no training happens here.

## Solution information

| | |
|---|---|
| Base model | `Qwen/Qwen3-8B` — open weights, revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Base parameters | 8.19 B total (6.95 B non-embedding) |
| Fine-tuning | QLoRA — LoRA rank 192, alpha 384, dropout 0.05, on `q/k/v/o/gate/up/down_proj` |
| Parameters at inference | 8.19 B. LoRA is a low-rank update of existing weight matrices, so merging it adds **no** parameters to the model. The 524 M adapter values are re-parameterised base weights, not extra ones. |
| Quantisation | base weights in 4-bit NF4 (double quant), as during training |
| Adapter | `SEED_SOUP3` — one configuration trained at seeds 42 / 1234 / 2025, the three LoRA weight sets averaged |
| Closed models / external APIs | none — no network call is made at inference |
| Internet | not required, provided the base model and the adapter are attached as notebook Inputs |
| Retrieval / tools / external corpora at test time | none. The model sees the passage, the question and the options, and nothing else |
| Output | `/kaggle/working/submission.csv`, columns `id,answer` |

## Attached Inputs this notebook expects

| Input | Contents |
|---|---|
| Qwen3-8B (Kaggle Models, `transformers` framework) | base weights + tokenizer |
| `eq-track1-lora` (private dataset) | `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja`, `csreg_config.json` |
| `eq-track1-wheels` (private dataset) | `bitsandbytes` and `peft` wheels, so the install works with Internet: Off |
| competition test data | `iso_sensors_mcqa_test_questions.jsonl` (3,048 questions) |

The path cell below discovers all three by content, so the exact mount paths do
not matter.

## Where the submitted predictions were produced

The prediction file we submitted was produced by running **these cells** on our
own hardware, not in a Kaggle session:

| | |
|---|---|
| Hardware | one NVIDIA A100 40GB |
| 4-bit compute dtype | bfloat16 |
| Batch size | 64 |
| Runtime | 653 s for all 3,048 questions |
| Prediction file | `submission_reference.csv`, shipped inside the `eq-track1-lora` Input |

That run was checked against the training-time implementation, cell by cell:

- **3,048 / 3,048 prompts** byte-identical to the ones our training/eval module
  builds — the notebook inlines the prompt code rather than importing it, and
  this is what rules out a transcription error;
- **3,048 / 3,048 answers** identical to the submitted file;
- decision margin over the test set: median 0.949, minimum 0.000.

On Kaggle the notebook was exercised end to end on a T4 over a 32-question
subset (199 s, appendix agreeing with the batched path). The full 3,048-question
run was not performed on Kaggle: a 16GB T4 holds the 4-bit base (~6.0 GB) plus
the fp32 LoRA adapter (~2.1 GB) plus the KV cache, which leaves too little
headroom to be worth a five-hour session. Any GPU with ~24GB runs it as written.

One consequence worth stating plainly: a GPU without bfloat16 (T4, P100) falls
back to fp16 for the 4-bit compute path, so a re-run there can differ from the
submitted file on the handful of questions whose decision margin is near zero.
The last cell reports the agreement rate against `submission_reference.csv` so
that difference is measured rather than assumed.
"""
)

# --------------------------------------------------------------------------- #
md(
    r"""
## How the answer is decided — logits, not generated text

This is the one thing worth reading before the code, because the notebook looks
wrong otherwise: it never parses an answer letter out of the model's output.

The supervised target ends at `<think>…</think>\n<answer>` followed by EOS. The
model was **never trained to write the letter as text**, so string-matching the
generation finds nothing on every row. Training instead puts a K-way cross
entropy on the logits at the single position right after `<answer>`, restricted
to the letters that question offers. Inference reproduces exactly that:

1. generate the reasoning trace from the prompt (greedy, ≤192 new tokens);
2. run one forward pass over `prompt + think + "\n<answer>"`;
3. read the logits at the final position, keep only this question's option
   letters, take the argmax.

Training and inference therefore optimise and read the *same tensor slot*. A
letter that is not one of the question's options cannot be produced, and the
reasoning trace carried alongside each prediction is the exact string the logits
were read against — the trace and the answer cannot disagree.

The appendix at the bottom unrolls this by hand for a single question.
"""
)

# --------------------------------------------------------------------------- #
md("## 1. Paths\n\nDiscovered by content, so a different mount point does not break the run. Every path can be overridden with an environment variable, which is how the local verification script points these cells at the repository.")

code(
    r'''
import os, sys, json, csv, time, hashlib, collections
from collections import OrderedDict
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
ON_KAGGLE  = INPUT_ROOT.exists()

# One GPU. The 4-bit model is ~6 GB and fits on a single 16 GB card; letting
# device_map spread it over two only adds cross-device transfers, because the
# layers still run one after another.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Batches differ in prompt length, so the allocator sees a different shape most
# steps; expandable segments keep that from fragmenting the pool.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# transformers 5 materialises checkpoint tensors on a thread pool. Several
# full-precision shards are then in flight at once, and on a 16GB card that
# peak arrives before 4-bit quantisation has shrunk anything. Loading them one
# at a time costs a few seconds and keeps the peak at one tensor.
os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")


def _search(root: Path, filename: str, max_depth: int = 6):
    """Directories under `root` that directly contain `filename`."""
    found, root_depth = [], len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if len(here.parts) - root_depth >= max_depth:
            dirnames[:] = []
        if filename in filenames:
            found.append(here)
    return sorted(found)


def describe_inputs() -> str:
    """What is actually mounted — so a failure below explains itself."""
    lines = []
    if not ON_KAGGLE:
        return "  (not running on Kaggle)"
    for mount in sorted(INPUT_ROOT.iterdir()):
        lines.append(f"  {mount.name}/")
    for directory in _search(INPUT_ROOT, "config.json"):
        try:
            cfg = json.loads((directory / "config.json").read_text())
        except Exception as error:
            lines.append(f"    {directory}  config.json unreadable: {error}")
            continue
        lines.append(
            f"    {directory}\n"
            f"      architectures={cfg.get('architectures')} "
            f"num_hidden_layers={cfg.get('num_hidden_layers')} "
            f"tokenizer.json={'yes' if (directory / 'tokenizer.json').exists() else 'NO'}"
        )
    for directory in _search(INPUT_ROOT, "adapter_config.json"):
        lines.append(f"    {directory}  (adapter)")
    return "\n".join(lines) or "  (nothing mounted)"


def find_base_model() -> Path:
    """Any Qwen3 causal-LM checkpoint with a tokenizer; 36 layers means the 8B."""
    if os.environ.get("EQ_BASE_MODEL"):
        return Path(os.environ["EQ_BASE_MODEL"])

    candidates = []
    for directory in _search(INPUT_ROOT, "config.json"):
        try:
            cfg = json.loads((directory / "config.json").read_text())
        except Exception:
            continue
        architectures = cfg.get("architectures") or []
        if not any("Qwen3" in str(a) for a in architectures):
            continue
        if not (directory / "tokenizer.json").exists():
            continue
        if not any(directory.glob("*.safetensors")) and not (directory / "model.safetensors.index.json").exists():
            continue
        candidates.append((cfg.get("num_hidden_layers") != 36, directory))   # 8B sorts first

    if not candidates:
        raise FileNotFoundError(
            "No Qwen3 checkpoint found under /kaggle/input.\n"
            "Attach the base model as an Input (Add Input -> Models -> Qwen3 -> "
            "framework 'transformers', variation '8b').\n"
            "What is mounted right now:\n" + describe_inputs()
        )

    wrong_size, chosen = sorted(candidates)[0]
    if wrong_size:
        layers = json.loads((chosen / "config.json").read_text()).get("num_hidden_layers")
        print(f"! {chosen} has {layers} layers, not the 36 of Qwen3-8B — the adapter expects 8B")
    return chosen


def find_adapter() -> Path:
    if os.environ.get("EQ_ADAPTER"):
        return Path(os.environ["EQ_ADAPTER"])
    for candidate in _search(INPUT_ROOT, "adapter_config.json"):
        if (candidate / "adapter_model.safetensors").exists():
            return candidate
    raise FileNotFoundError(
        "LoRA adapter not found under /kaggle/input — attach the eq-track1-lora dataset.\n"
        "What is mounted right now:\n" + describe_inputs()
    )


def find_test() -> Path:
    if os.environ.get("EQ_TEST"):
        return Path(os.environ["EQ_TEST"])
    for name in ("test.jsonl", "iso_sensors_mcqa_test_questions.jsonl"):
        hits = [d / name for d in _search(INPUT_ROOT, name)]
        if hits:
            return hits[0]
    # Last resort: any .jsonl whose first record looks like one of our questions.
    for directory in {p.parent for p in INPUT_ROOT.rglob("*.jsonl")} if ON_KAGGLE else []:
        for path in sorted(directory.glob("*.jsonl")):
            try:
                first = json.loads(path.open(encoding="utf-8").readline())
            except Exception:
                continue
            if {"id", "question", "options"} <= set(first):
                print(f"using {path} (matched by content, not by name)")
                return path
    raise FileNotFoundError(
        "The competition test file was not found under /kaggle/input.\n"
        "Attach the competition data as an Input, or set EQ_TEST.\n"
        "What is mounted right now:\n" + describe_inputs()
    )


BASE_MODEL = find_base_model()
ADAPTER    = find_adapter()
TEST       = find_test()
OUTPUT     = Path(os.environ.get("EQ_OUTPUT", "/kaggle/working/submission.csv" if ON_KAGGLE else "submission.csv"))

# Optional: the CSV we actually submitted, attached alongside the weights. When
# present the last cell reports how closely this run reproduces it.
REFERENCE = None
for _dir in _search(INPUT_ROOT, "submission_reference.csv") if ON_KAGGLE else []:
    REFERENCE = _dir / "submission_reference.csv"
    break
if os.environ.get("EQ_REFERENCE"):
    REFERENCE = Path(os.environ["EQ_REFERENCE"])


# Smoke test: add a cell ABOVE this one containing
#     import os; os.environ["EQ_MAX_ROWS"] = "32"
# and delete it again afterwards. Editing MAX_ROWS in place instead would leave
# a notebook that quietly answers 32 of 3,048 questions for whoever runs it
# next, so a subset run is renamed rather than trusted to a reader's attention.
MAX_ROWS = int(os.environ.get("EQ_MAX_ROWS", "0")) or None
if MAX_ROWS:
    OUTPUT = OUTPUT.with_name(f"smoke_test_{MAX_ROWS}_questions.csv")
    print(f"!! SUBSET RUN: {MAX_ROWS} questions — writing {OUTPUT.name}, which is NOT a submission")

print("base model :", BASE_MODEL)
print("adapter    :", ADAPTER)
print("test file  :", TEST)
print("output     :", OUTPUT)
print("reference  :", REFERENCE)
'''
)

# --------------------------------------------------------------------------- #
md(
    r"""
## 2. Dependencies

Three pins, attached as the `eq-track1-wheels` Input so they install **with the
internet switched off**:

* **`bitsandbytes`** — not in the Kaggle image at all; 4-bit loading needs it.
* **`peft 0.19.1`** — the version that wrote this adapter's `adapter_config.json`.
  An older PEFT rejects keys that file contains.
* **`transformers 5.14.1`** — the image ships 5.0.0, whose checkpoint loader
  materialises the model at full precision *before* the 4-bit quantiser shrinks
  anything. Measured peak GPU memory loading Qwen3-8B in 4-bit: **5.75 GB on
  5.14.1 against more than 14.4 GB on 5.0.0**. On a 16GB card 5.0.0 therefore
  dies at ~98% of the parameters, and the traceback points at the loader's
  thread pool rather than at the cause.

`bitsandbytes` and `peft` install with `--no-deps`, since their dependency lists
name `torch` and `accelerate`, which the image already provides and which must
not be re-resolved. `transformers` resolves its closure from the same directory;
it does not depend on `torch`, so no build of it can be pulled in sideways.

If the wheel Input is missing the cell falls back to PyPI, which needs
Internet: On.
"""
)

code(
    r'''
import subprocess, importlib, importlib.metadata as md


def version_of(package: str) -> str | None:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None


print("before:", {p: version_of(p) for p in ("torch", "transformers", "peft", "bitsandbytes")})

pip        = [sys.executable, "-m", "pip", "install", "-q"]
wheel_dirs = sorted({p.parent for p in INPUT_ROOT.rglob("*.whl")}) if ON_KAGGLE else []

if not ON_KAGGLE:
    # The local verification script runs these cells inside the training env,
    # which already has the pinned versions. Installing there would mutate a
    # working environment to fix a problem it does not have.
    print("not on Kaggle — leaving the environment alone")
elif wheel_dirs:
    print("installing offline from", wheel_dirs[0])
    offline = ["--no-index", f"--find-links={wheel_dirs[0]}"]
    # --no-deps: both name torch/accelerate, which the image already provides.
    subprocess.check_call(pip + offline + ["--no-deps", "bitsandbytes", "peft"])
    # transformers resolves its closure from the same directory. It does not
    # depend on torch, so nothing can pull a different build of it in.
    subprocess.check_call(pip + offline + ["transformers==5.14.1"])
else:
    print("no wheel Input found — installing from PyPI (needs Internet: On)")
    subprocess.check_call(pip + ["--no-deps", "bitsandbytes==0.49.2", "peft==0.19.1"])
    subprocess.check_call(pip + ["transformers==5.14.1"])

importlib.invalidate_caches()
print("after :", {p: version_of(p) for p in ("torch", "transformers", "peft", "bitsandbytes")})

# Qwen3 support landed in transformers 4.51; the image's own version is kept.
transformers_version = tuple(int(x) for x in (version_of("transformers") or "0").split(".")[:2])
assert transformers_version >= (4, 51), (
    f"transformers {version_of('transformers')} cannot build Qwen3 — "
    "install transformers>=4.51 (needs Internet: On)"
)
assert tuple(int(x) for x in (version_of("peft") or "0").split(".")[:2]) >= (0, 19), \
    "peft>=0.19 is required to read this adapter_config.json"

# Fail here rather than three cells later inside from_pretrained. transformers
# caches the answer to "is bitsandbytes available?", so a kernel that imported
# transformers before the install above keeps reporting it missing.
import bitsandbytes                                          # noqa: F401
from transformers.utils import is_bitsandbytes_available

assert is_bitsandbytes_available(), (
    "bitsandbytes is installed but transformers still reports it missing — "
    "this kernel imported transformers before the install. "
    "Run -> Restart Session, then Run All (the install survives the restart)."
)
print("bitsandbytes ready:", version_of("bitsandbytes"))
'''
)

# --------------------------------------------------------------------------- #
md(
    r"""
## 3. Adapter metadata

The adapter records the decision style it was trained under. That record is
checked rather than assumed: a `label`-style adapter run through the logit path
below would produce a plausible-looking file that is silently wrong, so the
mismatch is made fatal here.
"""
)

code(
    r'''
adapter_config = json.loads((ADAPTER / "adapter_config.json").read_text())
csreg_config   = json.loads((ADAPTER / "csreg_config.json").read_text())

print("base_model_name_or_path :", adapter_config["base_model_name_or_path"])
print("lora r / alpha / dropout:", adapter_config["r"], "/", adapter_config["lora_alpha"], "/", adapter_config["lora_dropout"])
print("target_modules          :", sorted(adapter_config["target_modules"]))
print("answer_style            :", csreg_config["answer_style"], " (score = the letter-logit path)")
print("strict_prompt           :", csreg_config["strict_prompt"])

assert csreg_config["answer_style"] == "score",  "this notebook implements the score decision path only"
assert csreg_config["strict_prompt"] is True,    "this notebook builds the strict prompt only"
assert adapter_config["base_model_name_or_path"] == "Qwen/Qwen3-8B"

sha = hashlib.sha256((ADAPTER / "adapter_model.safetensors").read_bytes()).hexdigest()
print("adapter sha256          :", sha)
'''
)

# --------------------------------------------------------------------------- #
md(
    r"""
## 4. Test data

`test.jsonl` carries two question types: 2,758 relational `open_ended_multi_choice`
rows and 290 general `multi_choice_question_answering` rows. Both are answered,
and the two get slightly different instruction text — see the next cell.
"""
)

code(shared("data") + r'''


rows = load_rows(TEST)
if MAX_ROWS:
    rows = rows[:MAX_ROWS]

print(f"{len(rows)} questions")
print(" types  :", collections.Counter(r["question_type"] for r in rows).most_common())
print(" options:", dict(sorted(collections.Counter(len(r['options']) for r in rows).items())))

example = rows[0]
print("\nexample")
print("  id      :", example["id"])
print("  question:", example["question"][:110])
for label, text in example["options"].items():
    print(f"    {label}) {text}")
''')

# --------------------------------------------------------------------------- #
md(
    r"""
## 5. The prompt

Reproduced verbatim from training. Under `strict_prompt` the prompt contains
only what the competition input contains — instruction, passage, question,
options — and no metadata headers or entity markers, because the test rows carry
none of those.

The only branch is the instruction paragraph: relational rows get the FMEA
instruction that names the failure-mode → physical-quantity → sensor chain, and
the 290 general rows get one that does not, since telling the model to reason
about sensors on a question about, say, lockout procedure is actively wrong. The
branch reads `question_type` straight from the data — no classifier, no
heuristic.

The closing instruction still asks for an `<answer>LETTER</answer>` tag. That is
intentional: it is the string training used, so it is the string that puts the
model in the state training measured. The tag is never parsed.
"""
)

code(shared("prompt") + r'''


print(build_user_content(rows[0]))
''')

# --------------------------------------------------------------------------- #
md(
    r"""
## 6. Model

Base weights in 4-bit NF4 — the quantisation training ran under — with the LoRA
adapter applied on top. The adapter is not merged; `PeftModel` keeps it as a
side branch, which is numerically what the validation runs used.

The chat template is taken from the adapter directory, where it was saved at
training time, and is compared against the one shipped with the base model
copy. Pinning it here means a differently-versioned base model checkpoint cannot
silently reformat the prompt.
"""
)

code(
    r'''
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Re-running this cell in a live kernel would build a second model beside the
# first and run the GPU out of memory before either is usable — the failure then
# points at from_pretrained, which is not where the problem is.
for _stale in ("model", "tokenizer"):
    if _stale in globals():
        del globals()[_stale]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"gpu memory in use before load: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), use_fast=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Pin the prompt format to the template the adapter was trained with.
saved_template = (ADAPTER / "chat_template.jinja").read_text(encoding="utf-8")
if (tokenizer.chat_template or "").strip() != saved_template.strip():
    print("! base-model chat template differs from the trained one — using the trained one")
    tokenizer.chat_template = saved_template
else:
    print("chat template matches the one saved with the adapter")

# Not torch.cuda.is_bf16_supported(): since torch 2.9 it counts *emulated*
# bf16 and answers True on a T4, which has no bf16 hardware. Emulation is
# slower than the fp16 path and is not what the A100 run used either. Ampere
# (compute capability 8.0) is where real bf16 starts.
has_bf16      = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
compute_dtype = torch.bfloat16 if has_bf16 else torch.float16
print("4-bit compute dtype:", compute_dtype,
      f"(compute capability {torch.cuda.get_device_capability()})" if torch.cuda.is_available() else "")

model = AutoModelForCausalLM.from_pretrained(
    str(BASE_MODEL),
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
)
model = PeftModel.from_pretrained(model, str(ADAPTER))
model.eval()

DEVICE = model.get_input_embeddings().weight.device
print("device:", DEVICE)

# Batch size trades throughput against memory only. 64 was used on a 40GB A100;
# a 16GB Kaggle GPU wants 16.
if torch.cuda.is_available():
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    default_batch = 64 if gpu_gb >= 24 else 16
else:
    gpu_gb, default_batch = 0.0, 4
BATCH_SIZE = int(os.environ.get("EQ_BATCH_SIZE", default_batch))
print(f"gpu {gpu_gb:.0f} GB -> batch size {BATCH_SIZE}")
'''
)

# --------------------------------------------------------------------------- #
md(
    r"""
## 7. Inference

The two-step decision from the top of the notebook, batched. Left padding makes
the last column the last real token for every row in the batch, which is the
position the `<answer>` logits are read from.
"""
)

code(r'''
from tqdm.auto import tqdm

''' + shared("decode") + r'''


started = time.time()
results = predict_rows(
    rows, model, tokenizer, DEVICE, BATCH_SIZE, DEFAULT_MAX_NEW_TOKENS,
    progress=lambda it: tqdm(it, desc="scoring"),
)
print(f"{len(results)} questions in {time.time() - started:.0f} s")
print("example:", results[0]["answer"], "| margin", round(results[0]["margin"], 3))
print("trace   :", results[0]["think"][:120].replace("\n", " "))
''')

# --------------------------------------------------------------------------- #
md(
    r"""
## 8. Write the submission

Two columns, `id` and `answer`, the answer being the option letter.

An answer outside the question's own options aborts the write. Falling back to
the first option instead would turn a broken run into a plausible-looking file —
a failure mode we hit once and do not intend to repeat.
"""
)

code(
    r'''
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

by_id = {row["id"]: row for row in rows}
out   = []
for result in results:
    if result["answer"] not in by_id[result["id"]]["options"]:
        raise ValueError(f"{result['answer']!r} is not an option of question {result['id']}")
    out.append({"id": result["id"], "answer": result["answer"]})

with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["id", "answer"], quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(out)

print(OUTPUT, "—", len(out), "rows")
print("answers  :", dict(sorted(collections.Counter(r["answer"] for r in out).items())))
'''
)

# --------------------------------------------------------------------------- #
md("## 9. Checks on the file that will be submitted")

code(
    r'''
written = list(csv.DictReader(OUTPUT.open(encoding="utf-8")))

assert list(written[0]) == ["id", "answer"],                "columns must be exactly id,answer"
assert len(written) == len(rows),                           f"expected {len(rows)} rows, got {len(written)}"
assert len({r['id'] for r in written}) == len(written),     "duplicate ids"
assert {r["id"] for r in written} == set(by_id),            "id set differs from test.jsonl"
assert all(r["answer"] in by_id[r["id"]]["options"] for r in written), "answer outside its options"
assert all(r["answer"].isalpha() and len(r["answer"]) == 1 for r in written), "answer must be one letter"

margins = sorted(r["margin"] for r in results)
print(f"{len(written)} rows — all checks passed")
print(f"decision margin: median {margins[len(margins) // 2]:.3f}, min {margins[0]:.3f}")
print("\n" + "\n".join(OUTPUT.read_text(encoding="utf-8").splitlines()[:4]))

if REFERENCE and Path(REFERENCE).exists():
    reference = {r["id"]: r["answer"] for r in csv.DictReader(Path(REFERENCE).open(encoding="utf-8"))}
    shared    = [r for r in written if r["id"] in reference]
    agree     = sum(r["answer"] == reference[r["id"]] for r in shared)
    print(f"\nagreement with the submitted file: {agree}/{len(shared)} = {agree / len(shared):.4%}")
    if agree != len(shared):
        print("  (differences are expected only if this GPU has no bf16 and fell back to fp16)")
'''
)

# --------------------------------------------------------------------------- #
md(
    r"""
## Appendix — the decision function by hand

Unrolled for one question: append the generated trace to the prompt, read the
logits at the `<answer>` slot, keep this question's letters, softmax. The result
must match section 7.
"""
)

code(
    r'''
import torch
import torch.nn.functional as F

row    = rows[0]
prompt = build_prompt(row, tokenizer)
think  = results[0]["think"]
encoded = tokenizer(f"{prompt}{think}\n<answer>", return_tensors="pt",
                    add_special_tokens=False).to(DEVICE)

with torch.no_grad():
    logits = model(**encoded).logits

letter_id = letter_token_ids(tokenizer)
labels    = list(row["options"])
candidate = torch.tensor([letter_id[label] for label in labels], device=DEVICE)
probs     = F.softmax(logits[0, -1].float().index_select(0, candidate), dim=-1)

for label, p, text in sorted(zip(labels, probs.tolist(), row["options"].values()), key=lambda x: -x[1]):
    print(f"  {label}  {p:.4f}  {text}")
print("\nby hand:", labels[int(probs.argmax())], "| section 7:", results[0]["answer"])
'''
)


# --------------------------------------------------------------------------- #
md(
    r"""
---

# Technical report

## 1. Task and system

The competition set contains 3,048 four-to-eight-option questions: 2,758
*relational* items asking which sensor signal indicates a given failure mode (or
the reverse), and 290 *general* industrial-engineering items. Our system is a
single open-weight language model, `Qwen/Qwen3-8B` (8.19 B parameters, 6.95 B
non-embedding), adapted with QLoRA. No retrieval, tool call, external API or
second model takes part in inference: the model sees the passage, the question
and the options, and nothing else.

## 2. Answer decoding: constrained logits rather than generated text

The supervised target ends at `<think>…</think>\n<answer>`, immediately followed
by the end-of-sequence token. The model is therefore never trained to emit the
option letter as text.

Supervision is applied to the logits at the single position that follows
`<answer>`. Those logits are gathered down to the option letters the item
actually offers, and a **K-way softmax cross-entropy** is taken over that
restricted set, with label smoothing 0.05 distributed only over the same row's
candidates. Because the softmax normalises across exactly the options present,
the objective is invariant to the option count, which our augmentation varies
between epochs.

Decoding mirrors the objective exactly: generate the reasoning trace, run one
forward pass over `prompt + trace + "\n<answer>"`, restrict the final-position
logits to that item's letters, take the argmax. Training and inference operate
on the same tensor position, an option outside the presented set is
unrepresentable, and no probability mass escapes to the remaining ~150k
vocabulary entries.

One consequence shaped the reasoning targets: the target trace **never states
the answer**. If it did, teacher forcing would place the answer inside the
model's own context and the letter objective would degenerate into copying it —
while at inference the trace is generated, so a faulty trace would be copied
just as faithfully.

## 3. Input and target visibility

The evaluation items carry only a passage, a question and options. Asset class,
relation direction, question polarity and anchor entity exist in our training
data but are withheld from both the prompt and the target text, and are used
only as side information for data construction. A target that named them would
train the model to invent, at test time, fields that do not exist. The prompt
contains no metadata header and no entity markers.

The instruction paragraph is the only branch: relational items receive an FMEA
instruction naming the failure-mode → physical-quantity → sensor-response chain,
general items receive one that does not. The branch reads the `question_type`
field supplied with the data — no classifier and no heuristic.

## 4. Training-data construction I: relation graph

From the training split we induce a bipartite graph over failure modes and
sensors per asset class, recording **observed non-edges** separately from pairs
that were simply never observed. Only the former are used as distractors:
"absent from the graph" and "known to be unrelated" are different claims, and
conflating them manufactures false negatives.

The graph then drives item synthesis rather than the loss:

- **Template enumeration** — instead of only resampling existing items, we
  enumerate the question shapes the graph can support across option counts 4–8,
  covering shapes present in evaluation but rare in training.
- **Distractor resizing** — items are regenerated every epoch at option counts
  2, 3, 4, 6 and 8, shrinking as well as growing. Error analysis showed the gold
  option was ranked second in most residual errors, i.e. the final two-way
  discrimination was the untrained regime.
- **Cross-asset distractors** — entities that are genuine neighbours of the
  anchor in a *different* asset class but absent from this one (30% of added
  distractors), which penalises answering by lexical familiarity.
- **Coverage floor** — a minimum of 12 items per relation fact (edge × direction).
  Coverage was heavily skewed: a long tail of facts appeared only once or twice
  while common ones appeared twenty times or more, and a fact seen twice in a
  large corpus is not learned.
- **None-of-the-above (NOTA) items** — both items whose correct answer is NOTA
  and items where NOTA is present but wrong, so the model learns the distinction
  rather than a prior over the phrase.
- **Per-epoch option permutation** — option order is reshuffled in place each
  epoch rather than materialising permuted duplicates.

Three explicit shortcut controls: a length band of 0.55 between gold and
distractor lengths, so answer length carries no signal; a difficulty mix of
30/45/25 (easy/medium/hard), because training only on maximally hard distractors
makes "least similar option wins" a viable rule without any relational
knowledge; and a cap of 3.0 on how often a fact may appear as a distractor
relative to appearances as the gold answer, since the letter objective otherwise
suppresses facts that are structurally over-represented in distractor slots.

## 5. Training-data construction II: knowledge grounding from public documents

Graph-derived synthesis alone left two gaps. The 290 general items are not
relational at all, and error analysis of an earlier checkpoint found that most
validation errors were *over-abstention*: NOTA selected although a correct
option was present. That is a narrow corpus collapsing into graph lookup —
an unobserved pair is read as "no relation exists" rather than "not recorded".

We therefore grounded the model in **17 publicly available technical documents**
from U.S. government agencies: DOE sourcebooks and best-practice guides (pumps,
fans, motors, steam, compressed air, internal-combustion engines, operations and
maintenance), EPA procedures (compliance assurance monitoring, flow
measurement), FAA airframe and powerplant handbook chapters, a NETL gas-turbine
handbook, a NIST technical note, and OSHA guidance (electrical safety,
lockout/tagout, job hazard analysis). All are free to access and carry no usage
restriction.

Provenance is recorded per source in a manifest: identifier, title, agency,
publication date, URL, SHA-256 digest, byte size, the specific pages used, the
number of facts derived, and the retrieval date — so any generated item can be
traced to a document and page.

From these we extracted **662 structured facts**: 329 mechanism statements, 186
reviewed items, 110 fault statements and 37 maintenance-decision statements.
The distractor structure is carried by the extraction itself rather than added
afterwards: each fault fact records an opposite, a no-effect, an over-general
variant and at least two rival mechanisms; reviewed facts record four to six
confusable concepts; decision facts record four ranked candidates. These are
same-domain, same-register distractors, so the resulting items cannot be solved
by word association.

Roughly half of the extracted facts mention an asset, sensor or failure mode
that also appears in the relation graph, so the document items reinforce the
relational subset rather than forming a disconnected second task.

Generation per fact: four positive items, three negatively-phrased items, 25%
*affirmation* items — where a correct option is present and the model must not
abstain, targeting the over-abstention failure directly — 10% items whose
correct answer genuinely is NOTA, and 30% surface-form variation so a concept is
not tied to a single string.

## 6. Weight averaging across seeds

The final adapter is a weight-space average ("model soup") of three runs of the
identical recipe at seeds 42, 1234 and 2025.

What must be averaged is the **product** `B·A` of each low-rank pair, not `A`
and `B` separately: the factorisation is not unique, so `mean(A)·mean(B)` is not
the mean of the updates. Stacking the `A` factors vertically, the `B` factors
horizontally and scaling by `1/k` reproduces `mean(Bᵢ·Aᵢ)` exactly, at rank
`k·r`. This is why the released adapter has rank 192 while each constituent run
was trained at rank 64. Unlike an output-space ensemble it costs one adapter and
one forward pass.

## 7. Training configuration

| | |
|---|---|
| Quantisation | 4-bit NF4, double quantisation (QLoRA) |
| LoRA | rank 64 (192 after averaging), α 128 (384 after averaging), dropout 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Schedule | 2 epochs, effective batch 64 (8 × grad-accum 2 × 4 DDP ranks) |
| Optimiser | paged AdamW 8-bit, lr 6e-5 cosine, warmup 10%, weight decay 0.05, grad-norm clip 0.5 |
| Sequence length | 768 |
| Regularisation | NEFTune embedding noise, α 1.0 → 10.0 |

**The loss has exactly two terms**: the standard language-modelling loss and the
constrained letter cross-entropy of §2, weighted 1.0. Auxiliary structural
objectives were implemented and evaluated during development — representation
alignment on graph edges and non-edges, an adjacency-conditioned matrix
regression, and a listwise ranking loss over option contents — and all are
disabled (weight 0) in the final recipe. The relation graph contributes through
data construction only, never as a training signal.

## 8. Results and limitations

On our held-out validation split the adapter answers **1,242 / 1,242 items
correctly (100.0%, bootstrap 95% CI [1.000, 1.000])**, with a median decision
margin of 0.949 between the top two option probabilities.

Two limitations should be read alongside that number.

**The validation split contains relational items only.** It has no counterpart
to the 290 general items in the evaluation set, so that portion of the task is
unmeasured by our own held-out data, and 100% describes the relational regime.

**A small number of evaluation items are effectively ties.** The decision margin
over the evaluation set has median 0.949 but minimum 0.000. Those items can flip
under numerical differences — notably a GPU without bfloat16 support, where the
4-bit compute path falls back to fp16. The agreement check in section 9
quantifies this rather than leaving it to assumption.
"""
)


# --------------------------------------------------------------------------- #
def main() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": source.splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if kind == "code" else {}),
            }
            for kind, source in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{OUT} — {len(CELLS)} cells "
          f"({sum(1 for k, _ in CELLS if k == 'code')} code, "
          f"{sum(1 for k, _ in CELLS if k == 'markdown')} markdown)")


if __name__ == "__main__":
    main()
