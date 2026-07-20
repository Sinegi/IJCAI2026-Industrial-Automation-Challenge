from __future__ import annotations

import json
from pathlib import Path
from typing import Any, MutableMapping

from .core import structural_losses


def find_final_norm(model: Any) -> Any:
    """Find the decoder final norm for Qwen2/Qwen2.5 under PEFT wrappers."""
    paths = (
        "base_model.model.model.norm",
        "base_model.model.model.model.norm",
        "model.model.norm",
        "model.norm",
    )
    for path in paths:
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise AttributeError("Could not locate the decoder final norm; inspect the model architecture.")


def make_memory_efficient_csreg_trainer():
    from transformers import Trainer

    class MemoryEfficientCSRegTrainer(Trainer):
        """Trainer that captures only the final hidden state via a forward hook.

        ``output_hidden_states=True`` materializes every decoder layer and is wasteful
        on a 40 GB A100. The hook preserves the exact post-final-norm representation
        used by the LM head while keeping the rest of the training graph unchanged.
        """

        def __init__(
            self,
            *args: Any,
            lambda_signature: float = 0.02,
            lambda_reconstruction: float = 0.005,
            lambda_positive: float = 0.01,
            lambda_nonedge: float = 0.02,
            nonedge_margin: float = 0.15,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            # We return a mean micro-batch loss and intentionally do not normalize
            # by ``num_items_in_batch``.  Transformers v5 may otherwise infer that
            # the wrapped model accepts loss kwargs and apply incompatible gradient-
            # accumulation scaling.  Official Trainer docs recommend disabling this
            # flag when a custom compute_loss ignores num_items_in_batch.
            self.model_accepts_loss_kwargs = False
            self.lambda_signature = float(lambda_signature)
            self.lambda_reconstruction = float(lambda_reconstruction)
            self.lambda_positive = float(lambda_positive)
            self.lambda_nonedge = float(lambda_nonedge)
            self.nonedge_margin = float(nonedge_margin)
            self.latest_loss_components: dict[str, float] = {}
            self._component_sums: dict[str, float] = {}
            self._component_count = 0
            self._final_norm = find_final_norm(self.model)

        def compute_loss(
            self,
            model: Any,
            inputs: MutableMapping[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            anchor_span = inputs.pop("anchor_span")
            option_spans = inputs.pop("option_spans")
            option_mask = inputs.pop("option_mask")
            edge_targets = inputs.pop("edge_targets")
            direction_id = inputs.pop("direction_id")
            inputs.pop("asset_id", None)

            captured: dict[str, Any] = {}

            def hook(_module: Any, _args: Any, output: Any) -> None:
                captured["hidden"] = output[0] if isinstance(output, tuple) else output

            handle = self._final_norm.register_forward_hook(hook)
            try:
                outputs = model(**inputs, use_cache=False, output_hidden_states=False)
            finally:
                handle.remove()
            if "hidden" not in captured:
                raise RuntimeError("Final hidden-state hook did not fire")

            lm_loss = outputs.loss
            components = structural_losses(
                hidden=captured["hidden"],
                anchor_span=anchor_span,
                option_spans=option_spans,
                option_mask=option_mask,
                edge_targets=edge_targets,
                direction_id=direction_id,
                projector=model.csreg_projection,
                nonedge_margin=self.nonedge_margin,
            )
            total = (
                lm_loss
                + self.lambda_signature * components["signature"]
                + self.lambda_reconstruction * components["reconstruction"]
                + self.lambda_positive * components["positive_alignment"]
                + self.lambda_nonedge * components["nonedge_margin"]
            )
            self.latest_loss_components = {
                "lm": float(lm_loss.detach().float().cpu()),
                **{k: float(v.detach().float().cpu()) for k, v in components.items()},
                "total": float(total.detach().float().cpu()),
            }
            for key, value in self.latest_loss_components.items():
                self._component_sums[key] = self._component_sums.get(key, 0.0) + value
            self._component_count += 1
            return (total, outputs) if return_outputs else total

        def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
            # Report the mean over all compute_loss calls since the previous log.
            # The previous implementation exposed only the last micro-batch, which
            # made final metrics misleading when logging_steps exceeded total steps.
            if self._component_count > 0:
                for key, value in self._component_sums.items():
                    logs.setdefault(f"csreg/{key}", value / self._component_count)
                self._component_sums.clear()
                self._component_count = 0
            elif self.latest_loss_components:
                for key, value in self.latest_loss_components.items():
                    logs.setdefault(f"csreg/{key}", value)
            try:
                super().log(logs, start_time=start_time)
            except TypeError:  # transformers < 4.47 compatibility
                super().log(logs)

        def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
            import torch

            super().save_model(output_dir=output_dir, _internal_call=_internal_call)
            target = Path(output_dir or self.args.output_dir)
            target.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.csreg_projection.state_dict(), target / "csreg_projection.pt")
            (target / "csreg_config.json").write_text(
                json.dumps(
                    {
                        "lambda_signature": self.lambda_signature,
                        "lambda_reconstruction": self.lambda_reconstruction,
                        "lambda_positive": self.lambda_positive,
                        "lambda_nonedge": self.lambda_nonedge,
                        "nonedge_margin": self.nonedge_margin,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    return MemoryEfficientCSRegTrainer
