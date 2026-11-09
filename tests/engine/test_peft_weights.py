"""Tests for merging PEFT weights into HF exports."""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from torchturbo.engine.peft.weights import (
    compute_dora_magnitude,
    get_effective_weight,
    merge_lora_linear,
)


class LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def test_merge_lora_linear_matches_peft_dora_merge() -> None:
    model = get_peft_model(
        LinearModel(),
        LoraConfig(
            r=2,
            lora_alpha=3,
            target_modules=["linear"],
            use_dora=True,
        ),
    )
    layer = model.base_model.model.linear
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_()
        layer.lora_B["default"].weight.normal_()
        layer.lora_magnitude_vector["default"].weight.uniform_(0.5, 2.0)

    base_weight = layer.base_layer.weight.detach().clone()
    expected = layer.lora_variant["default"].merge_safe(
        layer, "default", base_weight.clone()
    )

    actual = merge_lora_linear(base_weight, layer, "weight")

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(base_weight, layer.base_layer.weight)


def test_compute_dora_magnitude_matches_peft_initialization() -> None:
    model = get_peft_model(
        LinearModel(),
        LoraConfig(
            r=2,
            lora_alpha=3,
            target_modules=["linear"],
            init_lora_weights=False,
            use_dora=True,
        ),
    )
    layer = model.base_model.model.linear

    actual = compute_dora_magnitude(layer, "default")

    torch.testing.assert_close(
        actual,
        layer.lora_magnitude_vector["default"].weight,
    )


def test_get_effective_weight_plain_linear_passthrough() -> None:
    linear = nn.Linear(3, 2, bias=False)
    assert get_effective_weight(linear) is linear.weight


def test_get_effective_weight_matches_manual_delta_and_merge() -> None:
    model = get_peft_model(
        LinearModel(),
        LoraConfig(r=2, lora_alpha=3, target_modules=["linear"]),
    )
    layer = model.base_model.model.linear
    # Non-zero B so the adapter actually changes the effective weight.
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_()
        layer.lora_B["default"].weight.normal_()

    base = layer.base_layer.weight
    weight_a = layer.lora_A["default"].weight
    weight_b = layer.lora_B["default"].weight
    expected = base + weight_b @ weight_a * layer.scaling["default"]

    torch.testing.assert_close(get_effective_weight(layer), expected)

    # Autograd must reach the adapter parameters through the returned tensor
    # (regression guard for weight-consuming forward paths that used to
    # read ``.weight`` directly).
    effective = get_effective_weight(layer)
    effective.sum().backward()
    assert layer.lora_A["default"].weight.grad is not None
    assert layer.lora_B["default"].weight.grad is not None

    # Merging folds the delta into the base weight: afterwards the effective
    # weight is exactly the merged base weight.
    before_merge = get_effective_weight(layer).detach().clone()
    model.merge_adapter()
    torch.testing.assert_close(layer.base_layer.weight, before_merge)
    assert get_effective_weight(layer) is layer.base_layer.weight
