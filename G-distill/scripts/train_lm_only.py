#!/usr/bin/env python3
"""Pure QLoRA SFT baseline for FailureSensorIQ.

This script intentionally does not load graph_bundle.json, does not attach the
CSReg projector, and does not compute any graph/structural loss.  It keeps the
same base model, LoRA modules, prompt template, assistant target, optimizer,
and training hyperparameters as the full CSReg run so that the comparison is
interpretable.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from fsiq_csreg.core import (  # noqa: E402
    NormalizedMCQ,
    build_assistant_target,
    build_marked_user_content,
    permute_options,
    seed_everything,
    stable_int,
)
from fsiq_csreg.public_data import read_prepared_rows  # noqa: E402


class FeatureDataset(Dataset):
    def __init__(self, features: list[dict[str, list[int]]]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


class CausalLMCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_length = int(math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of)
        pad_id = int(self.tokenizer.pad_token_id)
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        def pad(values: list[int], pad_value: int) -> list[int]:
            amount = max_length - len(values)
            if padding_side == "left":
                return [pad_value] * amount + list(values)
            return list(values) + [pad_value] * amount

        return {
            "input_ids": torch.tensor([pad(f["input_ids"], pad_id) for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in features], dtype=torch.long),
            "labels": torch.tensor([pad(f["labels"], -100) for f in features], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a graph-free, loss-free QLoRA SFT baseline")
    p.add_argument("--config", default="configs/a100_40gb.yaml")
    p.add_argument("--prepared-dir", default=None)
    p.add_argument(
        "--rows-file",
        default=None,
        help="Default: <prepared-dir>/public_single_normalized.jsonl",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--permutation-copies", type=int, default=0)
    p.add_argument(
        "--match-row-count-from",
        default=None,
        help="Generate additional non-graph option permutations until the row count matches this JSONL file.",
    )
    p.add_argument("--target-row-count", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None, help="Applied after augmentation; useful for smoke tests")
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--overwrite-output-dir", action="store_true")
    return p.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clone_permutation(row: NormalizedMCQ, seed: int, serial: int) -> NormalizedMCQ:
    clone = permute_options(row, seed)
    clone.id = f"{row.id}::nog_perm::{serial}"
    clone.metadata = dict(clone.metadata)
    clone.metadata.update({"ablation": "no_graph_permutation", "permutation_serial": serial})
    return clone


def build_no_graph_rows(
    base_rows: list[NormalizedMCQ],
    permutation_copies: int,
    target_count: int | None,
    seed: int,
) -> list[NormalizedMCQ]:
    base_rows = [r for r in base_rows if r.is_single_answer and r.anchor]
    if not base_rows:
        raise ValueError("No labeled single-answer rows with anchors were found")

    rows = list(base_rows)
    serial = 0
    for copy_idx in range(max(0, permutation_copies)):
        for row in base_rows:
            rows.append(
                clone_permutation(
                    row,
                    seed + copy_idx * 1_000_003 + stable_int(row.id),
                    serial,
                )
            )
            serial += 1

    if target_count is not None:
        if target_count < len(base_rows):
            raise ValueError(
                f"target_count={target_count} is smaller than the original row count={len(base_rows)}"
            )
        cycle = 0
        while len(rows) < target_count:
            for row in base_rows:
                if len(rows) >= target_count:
                    break
                rows.append(
                    clone_permutation(
                        row,
                        seed + 9_999_991 + cycle * 1_000_003 + stable_int(row.id),
                        serial,
                    )
                )
                serial += 1
            cycle += 1
    return rows


def tokenize_lm_row(row: NormalizedMCQ, tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    # Keep exactly the same prompt and target as the full method.  Only the graph
    # features and structural loss are removed.
    content, _ = build_marked_user_content(row)
    prompt_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    target = build_assistant_target(row)
    full_text = prompt_chat + target + (tokenizer.eos_token or "")

    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
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
    if all(x == -100 for x in labels):
        raise ValueError(f"Target was fully truncated for row {row.id}")
    return {
        "input_ids": list(encoded["input_ids"]),
        "attention_mask": list(encoded["attention_mask"]),
        "labels": labels,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    prepared_dir = Path(args.prepared_dir or cfg["data"]["prepared_dir"]).expanduser().resolve()
    rows_file = Path(args.rows_file or prepared_dir / "public_single_normalized.jsonl").expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.overwrite_output_dir and args.resume_from_checkpoint:
        raise ValueError("--overwrite-output-dir and --resume-from-checkpoint are mutually exclusive")
    if args.overwrite_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_count = args.target_row_count
    if args.match_row_count_from:
        match_rows = read_prepared_rows(Path(args.match_row_count_from).expanduser().resolve())
        target_count = len(match_rows)
        print(f"Matching training row count to {args.match_row_count_from}: {target_count}")

    original_rows = read_prepared_rows(rows_file)
    rows = build_no_graph_rows(
        original_rows,
        permutation_copies=args.permutation_copies,
        target_count=target_count,
        seed=seed,
    )
    if args.max_rows:
        rows = rows[: args.max_rows]

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"transformers={transformers.__version__}")
    print(
        f"NO-G baseline: source={rows_file} original={len(original_rows)} "
        f"training_rows={len(rows)} permutation_copies={args.permutation_copies}"
    )

    model_id = cfg["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    major_version = int(str(transformers.__version__).split(".", 1)[0])
    model_kwargs: dict[str, Any] = {
        "quantization_config": bnb_config,
        "device_map": {"": int(os.environ.get("LOCAL_RANK", "0"))},
        "trust_remote_code": bool(cfg["model"].get("trust_remote_code", False)),
        "attn_implementation": cfg["model"].get("attn_implementation", "sdpa"),
    }
    model_kwargs["dtype" if major_version >= 5 else "torch_dtype"] = compute_dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False
    try:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    max_length = int(cfg["training"]["max_length"])
    features: list[dict[str, list[int]]] = []
    skipped: list[tuple[str, str]] = []
    for row in tqdm(rows, desc="Tokenizing no-G training rows"):
        try:
            features.append(tokenize_lm_row(row, tokenizer, max_length=max_length))
        except Exception as exc:  # keep a complete audit trail
            skipped.append((row.id, repr(exc)))
    if not features:
        raise RuntimeError("No LM-only features were created")
    (output_dir / "skipped_rows.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    tcfg = cfg["training"]
    per_device_batch = int(tcfg["per_device_train_batch_size"])
    grad_accum = int(tcfg["gradient_accumulation_steps"])
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    epochs = float(args.epochs if args.epochs is not None else tcfg["epochs"])
    updates_per_epoch = math.ceil(len(features) / (per_device_batch * world_size * grad_accum))
    estimated_total_steps = max(1, math.ceil(updates_per_epoch * epochs))
    warmup_steps = int(tcfg.get("warmup_steps", 0))
    if warmup_steps <= 0:
        warmup_steps = max(1, round(estimated_total_steps * float(tcfg.get("warmup_ratio", 0.0))))

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": float(tcfg["learning_rate"]),
        "weight_decay": float(tcfg["weight_decay"]),
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": str(tcfg["lr_scheduler_type"]),
        "max_grad_norm": float(tcfg["max_grad_norm"]),
        "logging_steps": int(tcfg["logging_steps"]),
        "save_strategy": "steps",
        "save_steps": int(tcfg["save_steps"]),
        "save_total_limit": int(tcfg["save_total_limit"]),
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": str(tcfg["optim"]),
        "report_to": str(tcfg.get("report_to", "none")),
        "remove_unused_columns": False,
        "dataloader_num_workers": int(tcfg.get("dataloader_num_workers", 2)),
        "dataloader_pin_memory": True,
        "seed": seed,
        "data_seed": seed,
        "ddp_find_unused_parameters": False,
    }
    if args.max_steps is not None:
        training_kwargs["max_steps"] = int(args.max_steps)

    supported = set(inspect.signature(TrainingArguments).parameters)
    dropped = sorted(set(training_kwargs) - supported)
    if dropped:
        print(f"Ignoring unsupported TrainingArguments fields: {dropped}")
    train_args = TrainingArguments(**{k: v for k, v in training_kwargs.items() if k in supported})

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": train_args,
        "train_dataset": FeatureDataset(features),
        "data_collator": CausalLMCollator(tokenizer, pad_to_multiple_of=8),
    }
    trainer_init = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in trainer_init:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init:
        trainer_kwargs["tokenizer"] = tokenizer

    print(
        f"features={len(features)} updates_per_epoch={updates_per_epoch} "
        f"estimated_total_steps={estimated_total_steps} warmup_steps={warmup_steps}"
    )
    trainer = Trainer(**trainer_kwargs)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    manifest = {
        "experiment": "pure_qlora_no_graph",
        "uses_graph": False,
        "uses_graph_synthetic_data": False,
        "uses_structural_loss": False,
        "base_model": model_id,
        "source_rows_file": str(rows_file),
        "original_rows_loaded": len(original_rows),
        "training_rows": len(features),
        "permutation_copies_requested": args.permutation_copies,
        "target_row_count": target_count,
        "epochs": epochs,
        "max_steps": args.max_steps,
        "skipped_rows": len(skipped),
        "config": cfg,
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved pure QLoRA adapter to: {final_dir}")


if __name__ == "__main__":
    main()
