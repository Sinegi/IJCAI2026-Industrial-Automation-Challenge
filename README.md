# Industrial Automation Challenge
>🥈🎉 2nd Place — Industrial Automation Challenge, Track 1

This repository contains the inference code and submission artifacts for our **2nd-place solution** in Track 1 of the [Industrial Automation Challenge](https://sites.google.com/view/ai-industrial-challenge-ijcai/home).


Inference code and submission artifacts for our Track 1 entry: `Qwen/Qwen3-8B`
adapted with QLoRA, decoding the answer from constrained option-letter logits
rather than from generated text.

**Validation: 1,242 / 1,242 correct (100.0%, bootstrap 95% CI [1.000, 1.000])**,
median decision margin 0.949 — `reference/metrics_val.json`.

The method is written up in [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).

## Layout

```text
.
├── predictor.py                       # the inference implementation — one file, three callers
├── run.sh                             # check / smoke / test / val / direct
├── notebook/eq_track1_inference.ipynb # the Kaggle notebook (generated from predictor.py)
├── tools/build_notebook.py            # regenerates that notebook
├── adapter/                           # LoRA weights — fetched separately, see adapter/README.md
├── data/                              # test.jsonl (3,048 items), val.jsonl (1,242 items)
├── reference/                         # the submission we produced, and validation metrics
├── docs/TECHNICAL_REPORT.md
├── run.py, eval_framework.py, dataset_utils.py, metadata_config_*.json
└── requirements.txt
```

`run.py`, `eval_framework.py` and `dataset_utils.py` are the organizers' runner
from [AssetOpsBench](https://github.com/IBM/AssetOpsBench/tree/ijcai_2026_competition).
They are kept so the entry can be evaluated through the official harness;
`metadata_config_*.json` points that harness at `predictor.py:predict`.

### One implementation, not three

`predictor.py` is the only place the decision procedure exists. It is called by
the official harness (`predict(scenario)`), by its own command line, and by the
Kaggle notebook — which cannot import it, because it must run on Kaggle with no
repository attached. Instead `tools/build_notebook.py` splices the blocks marked
`# ===== SHARED:<name> =====` into the notebook cells verbatim, so the notebook
a reviewer reads is the same code the command line runs. Edit `predictor.py`,
then re-run:

```bash
python tools/build_notebook.py
```

## Running

Python ≥ 3.10 with a CUDA-enabled torch, and a GPU with ~10 GB free. Put the
adapter in `adapter/` first — see [`adapter/README.md`](adapter/README.md).

```bash
pip install -r requirements.txt

bash run.sh check    # resolve paths and print them — no model load, no GPU
bash run.sh smoke    # first 20 test items, end to end
bash run.sh test     # full test set  -> competition_results/submission.csv
bash run.sh val      # validation set -> competition_results/submission_val.csv
bash run.sh direct   # skip the harness -> results/submission.csv
```

`direct` is the faster path when you only want the CSV; it calls `predictor.py`
without the harness. Both routes run the identical decision procedure.

```bash
python predictor.py --data data/test.jsonl --output results/submission.csv
python predictor.py --limit 32            # smoke test
python predictor.py --describe            # resolved paths, no model load
```

The only download is the base model, `Qwen/Qwen3-8B`, pulled from the
HuggingFace hub on first run. Nothing else leaves the machine: no retrieval, no
tool call, no external API.

Useful environment variables: `EQ_ADAPTER`, `EQ_BASE_MODEL`, `EQ_BATCH_SIZE`,
`EQ_MAX_NEW_TOKENS`, `EQ_LOAD_IN_4BIT`. `predictor.py`'s docstring documents them.

## How the answer is decided

Worth knowing before reading the code, because it looks wrong otherwise: the
predictor never parses an answer letter out of the model's output.

The supervised target ends at `<think>…</think>\n<answer>` followed by EOS, so
the model was **never trained to write the letter as text**. Training puts a
K-way cross entropy on the logits at the single position after `<answer>`,
restricted to the letters that item offers. Inference reproduces exactly that:

1. generate the reasoning trace (greedy, ≤192 new tokens);
2. one forward pass over `prompt + trace + "\n<answer>"`;
3. restrict the final-position logits to this item's letters, take the argmax.

Training and inference read the same tensor position. An option outside the
presented set cannot be produced, and the trace stored with each prediction is
the exact string the logits were read against, so the two cannot disagree.

## Results

|                           |                                                         |
| ------------------------- | ------------------------------------------------------- |
| Validation accuracy       | **100.0%** (n = 1,242, bootstrap 95% CI [1.000, 1.000]) |
| Median decision margin    | 0.949                                                   |
| Reference prediction file | `reference/submission.csv` (3,048 rows)                 |
| Runtime                   | 653 s for 3,048 items on one A100 40GB, batch 64, bf16  |


