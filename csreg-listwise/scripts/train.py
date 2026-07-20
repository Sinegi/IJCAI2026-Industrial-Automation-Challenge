#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsiq_csreg.core import (  # noqa: E402
    CSRegDataCollator,
    attach_csreg_projector,
    load_graph_bundle,
    seed_everything,
    set_answer_style,
    tokenize_training_row,
)
from fsiq_csreg.public_data import read_prepared_rows  # noqa: E402
from fsiq_csreg.trainer import make_memory_efficient_csreg_trainer  # noqa: E402


class FeatureDataset(Dataset):
    def __init__(self, features: list[dict]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return self.features[index]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DeepSeek-R1-Distill-Qwen-7B with QLoRA + CSReg")
    p.add_argument("--config", default=str(ROOT / "configs" / "a100_40gb.yaml"))
    p.add_argument("--prepared-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--max-rows", type=int, default=None, help="Smoke-test limit")
    p.add_argument("--disable-csreg", action="store_true")
    p.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Delete output-dir before training. Do not use together with --resume-from-checkpoint.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    # Option-Content Listwise SFT: pick the assistant/answer format before any
    # tokenization so training targets and inference prompts share one style.
    listwise_cfg = dict(cfg.get("listwise", {}) or {})
    answer_style = str(listwise_cfg.get("answer_style", "label"))
    set_answer_style(answer_style)

    prepared_dir = Path(args.prepared_dir or cfg["data"]["prepared_dir"]).expanduser().resolve()
    output_dir = Path(args.output_dir or cfg["training"]["output_dir"]).expanduser().resolve()
    if args.overwrite_output_dir and args.resume_from_checkpoint:
        raise ValueError("--overwrite-output-dir and --resume-from-checkpoint are mutually exclusive")
    if args.overwrite_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_prepared_rows(prepared_dir / "train_rows.jsonl")
    if args.max_rows:
        rows = rows[: args.max_rows]
    graphs, _ = load_graph_bundle(prepared_dir / "graph_bundle.json")

    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    print(f"transformers={transformers.__version__}")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
    # transformers v5 renamed ``torch_dtype`` to ``dtype``.  Keep one code path
    # for both v4 and v5 because server environments often already contain v5.
    major_version = int(str(transformers.__version__).split(".", 1)[0])
    model_load_kwargs = dict(
        quantization_config=bnb_config,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
    )
    model_load_kwargs["dtype" if major_version >= 5 else "torch_dtype"] = compute_dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_kwargs)
    model.config.use_cache = False
    try:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        # Older PEFT versions do not expose gradient_checkpointing_kwargs.
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
    model = attach_csreg_projector(
        model,
        rank=int(cfg["csreg"]["projector_rank"]),
        scale=float(cfg["csreg"]["projector_scale"]),
    )
    model.print_trainable_parameters()

    max_length = int(cfg["training"]["max_length"])
    features: list[dict] = []
    skipped = []
    for row in tqdm(rows, desc="Tokenizing training rows"):
        try:
            feature = tokenize_training_row(row, tokenizer, graphs, max_length=max_length)
            if feature["anchor_span"][0] < 0:
                skipped.append((row.id, "missing_anchor_span"))
                continue
            features.append(feature)
        except Exception as exc:
            skipped.append((row.id, repr(exc)))
    if not features:
        raise RuntimeError("No training features were produced. Inspect prepare_stats.json and skipped_rows.json")
    (output_dir / "skipped_rows.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    tcfg = cfg["training"]
    per_device_batch = int(tcfg["per_device_train_batch_size"])
    grad_accum = int(tcfg["gradient_accumulation_steps"])
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    updates_per_epoch = math.ceil(len(features) / (per_device_batch * world_size * grad_accum))
    estimated_total_steps = max(1, math.ceil(updates_per_epoch * float(tcfg["epochs"])))
    warmup_steps = int(tcfg.get("warmup_steps", 0))
    if warmup_steps <= 0:
        warmup_steps = max(1, round(estimated_total_steps * float(tcfg.get("warmup_ratio", 0.0))))
    print(
        f"features={len(features)} world_size={world_size} "
        f"updates_per_epoch={updates_per_epoch} estimated_total_steps={estimated_total_steps} "
        f"warmup_steps={warmup_steps}"
    )

    training_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=float(tcfg["epochs"]),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"]),
        warmup_steps=warmup_steps,
        lr_scheduler_type=str(tcfg["lr_scheduler_type"]),
        max_grad_norm=float(tcfg["max_grad_norm"]),
        logging_steps=int(tcfg["logging_steps"]),
        save_strategy="steps",
        save_steps=int(tcfg["save_steps"]),
        save_total_limit=int(tcfg["save_total_limit"]),
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=str(tcfg["optim"]),
        report_to=str(tcfg.get("report_to", "none")),
        remove_unused_columns=False,
        dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 2)),
        dataloader_pin_memory=True,
        seed=seed,
        data_seed=seed,
        ddp_find_unused_parameters=False,
    )
    # HF transformers v5 removed a number of TrainingArguments fields, including
    # overwrite_output_dir. Filter by the actual installed signature instead of
    # requiring users to downgrade a working CUDA environment.
    supported_training_args = set(inspect.signature(TrainingArguments).parameters)
    dropped_training_args = sorted(set(training_kwargs) - supported_training_args)
    if dropped_training_args:
        print(f"Ignoring unsupported TrainingArguments fields: {dropped_training_args}")
    train_args = TrainingArguments(
        **{k: v for k, v in training_kwargs.items() if k in supported_training_args}
    )

    lambdas = cfg["csreg"].copy()
    if args.disable_csreg:
        for key in ("lambda_signature", "lambda_reconstruction", "lambda_positive", "lambda_nonedge"):
            lambdas[key] = 0.0

    listwise_enabled = bool(listwise_cfg.get("enabled", False))
    lambda_listwise = float(listwise_cfg.get("lambda_listwise", 0.0)) if listwise_enabled else 0.0
    listwise_temperature = float(listwise_cfg.get("temperature", 0.1))
    print(
        f"answer_style={answer_style} listwise_enabled={listwise_enabled} "
        f"lambda_listwise={lambda_listwise} listwise_temperature={listwise_temperature}"
    )

    TrainerCls = make_memory_efficient_csreg_trainer()
    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=FeatureDataset(features),
        data_collator=CSRegDataCollator(tokenizer, pad_to_multiple_of=8),
        lambda_signature=float(lambdas["lambda_signature"]),
        lambda_reconstruction=float(lambdas["lambda_reconstruction"]),
        lambda_positive=float(lambdas["lambda_positive"]),
        lambda_nonedge=float(lambdas["lambda_nonedge"]),
        nonedge_margin=float(lambdas["nonedge_margin"]),
        lambda_listwise=lambda_listwise,
        listwise_temperature=listwise_temperature,
    )
    trainer_init_params = set(inspect.signature(TrainerCls.__init__).parameters)
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = TrainerCls(**trainer_kwargs)
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(output_dir / "final_adapter")
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    manifest = {
        "base_model": model_id,
        "training_rows": len(features),
        "skipped_rows": len(skipped),
        "max_length": max_length,
        "config": cfg,
        "csreg_disabled": bool(args.disable_csreg),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved adapter to: {output_dir / 'final_adapter'}")


if __name__ == "__main__":
    main()
