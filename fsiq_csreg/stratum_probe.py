"""Sub-epoch weakness measurement on variants the model has not been trained on.

Why not measure on validation
-----------------------------
Validation is the only unbiased yardstick left -- the public leaderboard has a
submission cap -- and feeding it back as a training signal destroys that. It is
also too small where it matters: the stratum with the one weakness that survived
six runs (power transformer rows that offer "None of the above") holds 96 rows,
so a mid-training reading carries a 95% interval of +-4.4pp against a measured
deficit of 8pp. Reweighting on that is automating the mistake that produced
run PAIR2.

Why not measure on the training loss
------------------------------------
It is saturated. PAIR finished at option_score 0.2374 against a label-smoothing
floor of 0.20 (K=4) to 0.26 (K=8), so the model sits at the floor everywhere and
no stratum stands out. There is nothing to weight by.

What this measures instead
--------------------------
The augmenter redraws every option set each epoch (11,002 variants per epoch on
run PAIR2, of which the specific option combinations are new). A variant the
sampler has not yet reached is a first exposure: the underlying G* fact was seen,
the option set was not. That is exactly validation's structure -- val asks about
facts the model trained on, in option sets it never saw -- so accuracy on held-out
variants is a val-shaped signal with no val contact, and we choose its size.

    val power-transformer + NOTA stratum:      96 rows   +-4.36pp
    probe drawn at 2,000 rows:                          +-0.96pp

Why sub-epoch
-------------
The runs are two epochs. An epoch-boundary hook fires once, which is a switch and
not a curriculum. The probe set is withheld from its epoch's training entirely,
so it can be scored at any step, and the weights it produces apply to every step
that follows. At 405 optimiser steps per epoch and a probe every 50, that is 8
updates per epoch rather than 1.
"""
from __future__ import annotations

import collections
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

STRATUM_UNKNOWN = "unknown"


def stratum_of(feature: Mapping[str, Any]) -> str:
    """Label a feature by the axes measured to matter, coarsely enough to fill.

    Three axes, in the order the evidence supports them:

      * NOTA presence -- the only deficit that held on all six runs (rows that
        offer it scored 93.6-96.1% against 98.4-99.3% for rows that do not).
      * option count, bucketed -- accuracy falls monotonically with it
        (100% at four options, 92.9% at eight on PAIR at epoch 2.0).
      * power transformer or not -- the one asset whose deficit is stable
        (0.36pp spread across six runs on 380 rows) rather than rotating, and
        its whole deficit lives in its NOTA rows (88.5% against 99.3%).

    Asset is deliberately NOT a free axis: fan, steam turbine and compressor look
    weak but their run-to-run spread (0.9-1.9pp on 80-110 rows) is as large as the
    deficit, so splitting on them would manufacture strata out of noise.
    """
    n_options = int(feature.get("n_options") or 0)
    if n_options >= 7:
        size = "7-8"
    elif n_options >= 6:
        size = "6"
    else:
        size = "4-5"
    nota = "nota" if feature.get("has_nota") else "plain"
    asset = "pt" if feature.get("asset") == "power transformer" else "other"
    return f"{nota}/{size}/{asset}"


def build_probe_indices(
    features: Sequence[Mapping[str, Any]],
    *,
    target_size: int,
    seed: int,
    min_per_stratum: int = 40,
) -> list[int]:
    """Choose withheld indices, balanced across strata rather than proportional.

    Proportional sampling would hand the largest share to the strata that are
    already at 99% and leave the ones under test too small to read. Each stratum
    gets an equal quota, capped by what it actually holds.
    """
    rng = random.Random(seed + 77_003)
    buckets: dict[str, list[int]] = collections.defaultdict(list)
    for index, feature in enumerate(features):
        buckets[stratum_of(feature)].append(index)
    if not buckets:
        return []

    quota = max(min_per_stratum, target_size // max(1, len(buckets)))
    chosen: list[int] = []
    for stratum in sorted(buckets):
        pool = list(buckets[stratum])
        rng.shuffle(pool)
        chosen.extend(pool[:quota])
    rng.shuffle(chosen)
    return chosen


def stratum_weights(
    accuracy: Mapping[str, float],
    counts: Mapping[str, int],
    *,
    floor: float = 1.0,
    ceiling: float = 3.0,
    min_count: int = 30,
    prior: Mapping[str, float] | None = None,
    momentum: float = 0.5,
) -> dict[str, float]:
    """Turn measured per-stratum accuracy into bounded per-row loss multipliers.

    ``ceiling`` exists because an unbounded 1/(1-acc) explodes as a stratum
    approaches perfection from the wrong side and would hand the whole gradient
    to whichever stratum happened to draw badly. ``min_count`` refuses to weight
    a stratum measured on too few rows to distinguish from noise -- the same
    guard the asset analysis needed. ``momentum`` blends with the previous
    weights so a single noisy reading cannot swing the curriculum.
    """
    raw: dict[str, float] = {}
    for stratum, acc in accuracy.items():
        if counts.get(stratum, 0) < min_count:
            raw[stratum] = floor
            continue
        # Error rate relative to the best stratum: a stratum that is already the
        # strongest gets the floor, and the weight grows with how far behind the
        # rest of the corpus a stratum sits.
        raw[stratum] = 1.0 + max(0.0, 1.0 - acc) * 10.0
    if not raw:
        return {}
    best = min(raw.values())
    scaled = {s: min(ceiling, max(floor, v / best if best > 0 else floor)) for s, v in raw.items()}
    if prior:
        scaled = {
            s: momentum * prior.get(s, floor) + (1.0 - momentum) * v for s, v in scaled.items()
        }
    return scaled


def score_probe(
    model: Any,
    collator: Any,
    features: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 32,
    device: Any = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Read letter-logit accuracy on withheld features, one forward per batch.

    No generation: the target ends at ``<answer>`` and the decision is the
    argmax over this row's letter tokens at that slot, so a single teacher-forced
    forward gives the same answer inference would. That is what makes a probe
    affordable inside the training loop -- roughly a minute for 2,000 rows.
    """
    import torch

    correct: collections.Counter = collections.Counter()
    total: collections.Counter = collections.Counter()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                chunk = list(features[start : start + batch_size])
                batch = collator(chunk)
                slot = batch.pop("answer_slot")
                letters = batch.pop("letter_token_ids")
                mask = batch.pop("letter_mask")
                gold = batch.pop("letter_gold_index")
                batch.pop("loss_weight", None)
                keep = {"input_ids", "attention_mask"}
                model_inputs = {
                    k: (v.to(device) if device is not None else v)
                    for k, v in batch.items()
                    if k in keep
                }
                logits = model(**model_inputs, use_cache=False).logits
                rows = torch.arange(logits.shape[0], device=logits.device)
                slot_logits = logits[rows, slot.to(logits.device)].float()
                picked = slot_logits.gather(1, letters.to(logits.device))
                neg_inf = torch.finfo(picked.dtype).min
                picked = torch.where(mask.to(logits.device), picked, torch.full_like(picked, neg_inf))
                predicted = picked.argmax(dim=-1).cpu()
                for feature, pred, target in zip(chunk, predicted.tolist(), gold.tolist()):
                    if target < 0:
                        continue
                    stratum = stratum_of(feature)
                    total[stratum] += 1
                    correct[stratum] += int(pred == target)
    finally:
        if was_training:
            model.train()
    return dict(correct), dict(total)


def summarise(
    correct_by_stratum: Mapping[str, int], count_by_stratum: Mapping[str, int]
) -> dict[str, dict[str, float]]:
    return {
        stratum: {
            "n": count_by_stratum[stratum],
            "accuracy": 100.0 * correct_by_stratum.get(stratum, 0) / count_by_stratum[stratum],
        }
        for stratum in sorted(count_by_stratum)
        if count_by_stratum[stratum]
    }
