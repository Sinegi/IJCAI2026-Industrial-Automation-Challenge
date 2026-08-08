# adapter/

The LoRA adapter is **not in this repository** — `adapter_model.safetensors`
alone is 2.0 GB, past what git should carry. Put its files here:

```text
adapter/
├── adapter_config.json
├── adapter_model.safetensors
├── chat_template.jinja
├── csreg_config.json
├── tokenizer.json
└── tokenizer_config.json
```

`predictor.py` reads `csreg_config.json` before loading anything and refuses to
run unless it says `answer_style: score` and `strict_prompt: true`. An adapter
trained under a different decoding convention would otherwise produce a
plausible-looking submission that is silently wrong.

## Getting the files

They are published as a Kaggle dataset, `eq-track1-lora`. With the Kaggle CLI
authenticated:

```bash
kaggle datasets download -d <owner>/eq-track1-lora -p adapter --unzip
```

Or point the predictor somewhere else without copying anything:

```bash
EQ_ADAPTER=/path/to/adapter bash run.sh test
```

## Identity

| | |
|---|---|
| Base model | `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Rank / alpha | 192 / 384 (three rank-64 runs averaged — see the technical report) |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Values | 523.8 M, fp32 |
| `adapter_model.safetensors` sha256 | `52502bab83e9953483a258637d22fb44b91b91fa475d1fccdefbbe74b8060a52` |

The digest above is what `reference/metrics_val.json` was measured on. Verify
with `sha256sum adapter/adapter_model.safetensors`.

`tokenizer.json` and `tokenizer_config.json` are carried so the directory is a
self-describing PEFT checkpoint; the predictor loads the tokenizer from the base
model and takes only `chat_template.jinja` from here, pinning the prompt format
to the one training used.
