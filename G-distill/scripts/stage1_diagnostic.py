#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsiq_csreg.core import (  # noqa: E402
    collect_frozen_entity_embeddings,
    fit_stage1_bhat,
    load_graph_bundle,
    seed_everything,
)
from fsiq_csreg.public_data import read_prepared_rows  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Optional frozen-embedding B-hat diagnostic")
    p.add_argument("--config", default=str(ROOT / "configs" / "a100_40gb.yaml"))
    p.add_argument("--prepared-dir", default=None)
    p.add_argument("--output", default="outputs/stage1_diagnostic.json")
    p.add_argument("--max-rows", type=int, default=1500)
    p.add_argument("--steps", type=int, default=400)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    prepared = Path(args.prepared_dir or cfg["data"]["prepared_dir"])
    rows = read_prepared_rows(prepared / "public_single_normalized.jsonl")
    rows = [r for r in rows if r.anchor and r.is_single_answer]
    graphs, _ = load_graph_bundle(prepared / "graph_bundle.json")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
    )
    embeddings = collect_frozen_entity_embeddings(
        rows,
        model,
        tokenizer,
        max_length=int(cfg["training"]["max_length"]),
        max_rows=args.max_rows,
    )
    results = []
    for asset, graph in graphs.items():
        bucket = embeddings.get(asset, {})
        result = fit_stage1_bhat(
            bucket.get("failure_modes", {}),
            bucket.get("sensors", {}),
            graph,
            steps=args.steps,
            seed=seed,
        )
        results.append(result)

    summary = {
        "base_model": cfg["model"]["base_model"],
        "max_rows": args.max_rows,
        "assets": results,
        "mean_edge_auroc": (
            sum(x["edge_auroc"] for x in results if "edge_auroc" in x)
            / max(1, sum("edge_auroc" in x for x in results))
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"mean_edge_auroc": summary["mean_edge_auroc"], "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
