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

They are published as a Kaggle dataset:

<https://www.kaggle.com/datasets/eunmochoi/eq-track1-lora>

The dataset is **private**, so the link resolves only for accounts it has been
shared with — competition reviewers are granted access alongside the notebook.
It is not a broken link.

Download it with `kagglehub`:

```bash
pip install kagglehub
KAGGLE_API_TOKEN=<your Kaggle API token> python - <<'PY'
import shutil
from pathlib import Path
import kagglehub

source = Path(kagglehub.dataset_download("eunmochoi/eq-track1-lora"))
target = Path("adapter"); target.mkdir(exist_ok=True)
for name in ("adapter_config.json", "adapter_model.safetensors", "chat_template.jinja",
             "csreg_config.json", "tokenizer.json", "tokenizer_config.json"):
    shutil.copy2(source / name, target / name)
print("adapter ready in", target)
PY
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
