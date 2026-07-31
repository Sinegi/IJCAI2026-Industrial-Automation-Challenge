# Industrial-Automation-Challenge


```text
.
├── eq_predictor.py           # our predictor behind the kit's predict(scenario) interface
├── run_eq.sh                 # one-command runner (check / smoke / test / val)
├── fsiq_csreg/               # our inference code (core.py + stratum_probe.py)
├── configs/eq_nota_gen.yaml  # model + inference config, and the training recipe
├── adapter/                  # the trained LoRA adapter (rank 64) -- 680 MB
├── data/                     # test.jsonl (3048 rows), val.jsonl (1242 rows)
├── reference/                # the submission CSV and val metrics we obtained
└── metadata_config_{test,val}.json
```

The only external download is the base model, `Qwen/Qwen3-8B`, pulled from the
HuggingFace hub on first run.

`adapter/` (TBD)

## Model

| | |
|---|---|
| base model | `Qwen/Qwen3-8B`, loaded 4-bit NF4 |
| adapter | LoRA r=64, α=128, on q/k/v/o/gate/up/down |
| answer_style | `score` → the answer is read from the letter logits, not generated text |
| validation accuracy | **99.28%** (n=1242, 95% CI 98.79–99.68) — `reference/metrics_val_NOTA_GEN.json` |

`answer_style: score` is the one thing worth knowing before reading the code.
The adapter is trained so that the K-way cross-entropy lands on the option-letter
logits at the `<answer>` slot; it is trained never to write the letter as prose.

`eq_predictor.py` reads that back from `adapter/csreg_config.json` and decodes
through the same tensor position the loss optimizes. Decoding it by parsing
generated text would answer from a different decision function entirely.

## Acknowledgements
https://github.com/IBM/AssetOpsBench/tree/ijcai_2026_competition

```bibtex
@misc{industrial-automation-challenge-track-1,
    author = {Prateek Biswas},
    title = {Industrial Automation Challenge - Track 1},
    year = {2026},
    howpublished = {\url{https://kaggle.com/competitions/industrial-automation-challenge-track-1}},
    note = {Kaggle}
}
