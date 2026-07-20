from collections import OrderedDict

import torch

from fsiq_csreg.core import (
    NormalizedMCQ,
    build_train_graphs,
    normalize_record,
    structural_losses,
)


def test_competition_val_schema():
    row = normalize_record(
        {
            "id": "v1",
            "question": "Which sensor is NOT relevant?",
            "options": {"A": "voltage", "B": "oil debris"},
            "answer": "B",
            "metadata": {
                "asset_class": "electric generator",
                "family": "negation_failure_to_sensor",
                "anchor": "Loss of output power phase",
                "n_options": 2,
            },
        }
    )
    assert row.asset == "electric generator"
    assert row.direction == "fm2sensor"
    assert row.polarity == "negative"
    assert row.anchor == "loss of output power phase"
    assert row.answer_label == "B"


def test_graph_semantics():
    pos = NormalizedMCQ(
        id="1",
        question="q",
        options=OrderedDict(A="vibration", B="light"),
        asset="pump",
        direction="fm2sensor",
        polarity="positive",
        anchor="bearing wear",
        correct_labels=["A"],
    )
    neg = NormalizedMCQ(
        id="2",
        question="q",
        options=OrderedDict(A="vibration", B="light"),
        asset="pump",
        direction="fm2sensor",
        polarity="negative",
        anchor="bearing wear",
        correct_labels=["B"],
    )
    graph = build_train_graphs([pos, neg])["pump"]
    assert ("bearing wear", "vibration") in graph.edges
    assert ("bearing wear", "light") in graph.observed_nonedges


def test_structural_loss_backward():
    b, t, h = 2, 12, 16
    hidden = torch.randn(b, t, h, requires_grad=True)
    anchor_span = torch.tensor([[1, 3], [2, 4]])
    option_spans = torch.tensor([[[4, 6], [7, 9]], [[4, 6], [7, 9]]])
    option_mask = torch.ones((b, 2), dtype=torch.bool)
    edge_targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    direction_id = torch.tensor([0, 1])
    projector = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.LayerNorm(h))
    losses = structural_losses(
        hidden,
        anchor_span,
        option_spans,
        option_mask,
        edge_targets,
        direction_id,
        projector,
    )
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert hidden.grad is not None
