"""LoRA correctness of DeepSeek V4's weight-consuming o-projection absorption.

The PostAttention absorption and the MTP head consume ``wo_a.weight``
through a grouped einsum instead of calling the module, so a plain
``.weight`` read silently drops the LoRA delta.  These tests guard the
``get_effective_weight`` fix: injected adapters must flow into the
absorption exactly like a merged weight, and adapter parameters must stay
in the autograd graph.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from torchturbo.models.deepseek_v4 import model as turbo_model
from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Config,
    PostAttentionOps,
)

from .test_mtp import _make_mtp_model


def _make_config() -> DeepseekV4Config:
    """Small config with o_groups=2 so the grouped absorption is exercised.

    Mirrors test_packed_inputs._make_model; o_groups=1 would degenerate
    the absorption into a plain linear call.
    """
    return DeepseekV4Config(
        vocab_size=32,
        hidden_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        q_lora_rank=4,
        qk_rope_head_dim=2,
        qk_nope_head_dim=2,
        v_head_dim=4,
        head_dim=4,
        o_groups=2,
        o_lora_rank=4,
        sliding_window=8,
        compress_ratios=[1],
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        max_position_embeddings=512,
        original_max_position_embeddings=512,
        use_cache=False,
    )


def _randomize_adapters(layer: nn.Module) -> None:
    """Make both LoRA factors non-zero (PEFT zeroes lora_B at init)."""
    with torch.no_grad():
        for adapter in layer.lora_A:
            layer.lora_A[adapter].weight.normal_()
            layer.lora_B[adapter].weight.normal_()


def _merged_linear(layer: nn.Module) -> nn.Linear:
    """A plain Linear holding base + delta of a single-adapter LoRA layer."""
    with torch.no_grad():
        merged = (
            layer.base_layer.weight
            + layer.lora_B["default"].weight
            @ layer.lora_A["default"].weight
            * layer.scaling["default"]
        )
    linear = nn.Linear(merged.shape[1], merged.shape[0], bias=False)
    linear.weight = nn.Parameter(merged)
    return linear


class _OProjection(nn.Module):
    """Holds wo_a/wo_b under their real attribute names for PEFT matching."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.wo_a = nn.Linear(
            config.num_attention_heads * config.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank, config.hidden_size, bias=False
        )


def test_mtp_wo_absorption_lora_keeps_adapter_in_graph():
    model = _make_mtp_model(num_stages=1)
    lora_model = get_peft_model(
        model, LoraConfig(r=2, lora_alpha=4, target_modules=["wo_a", "wo_b"])
    )
    wrapped = lora_model.base_model.model
    _randomize_adapters(wrapped.layers[0].attn.wo_a)
    _randomize_adapters(wrapped.layers[0].attn.wo_b)
    _randomize_adapters(wrapped.mtp[0].attn.wo_a)
    _randomize_adapters(wrapped.mtp[0].attn.wo_b)

    input_ids = torch.arange(1, 11).unsqueeze(0)
    outputs = lora_model(input_ids=input_ids)
    loss = outputs.mtp_logits[0].square().sum()
    loss.backward()

    # Regression guard: the MTP absorption also used to read ``wo_a.weight``
    # (base only), keeping the adapter out of the autograd graph.
    mtp_wo_a = wrapped.mtp[0].attn.wo_a
    grad_a = mtp_wo_a.lora_A["default"].weight.grad
    grad_b = mtp_wo_a.lora_B["default"].weight.grad
    assert grad_a is not None and grad_a.abs().sum() > 0
    assert grad_b is not None and grad_b.abs().sum() > 0


def test_post_attention_absorption_lora_matches_merged_weight():
    config = _make_config()
    lora_holder = get_peft_model(
        _OProjection(config),
        LoraConfig(r=2, lora_alpha=4, target_modules=["wo_a", "wo_b"]),
    )
    wrapped = lora_holder.base_model.model
    _randomize_adapters(wrapped.wo_a)
    _randomize_adapters(wrapped.wo_b)

    # Pin the Turbo patched forward (model.py post_attention_forward) so
    # the test is independent of the process-wide patch state that other
    # tests may already have applied.  patch_deepseek_v4_attention_for_sp is
    # the class-level patch on_parallelized installs in production; without
    # it this test depends on another test having patched the class first.
    turbo_model.TurboDeepseekV4ForCausalLM.patch_deepseek_v4_attention_for_sp(
        sp_group=None, sp_size=1, sp_rank=0
    )
    ops = PostAttentionOps(config)

    torch.manual_seed(11)
    bsz, seqlen = 2, 3
    o = torch.randn(bsz, seqlen, config.num_attention_heads, config.head_dim)
    # Identity rotation keeps the comparison focused on the wo_a absorption.
    freqs_cis = torch.ones(
        config.max_position_embeddings,
        config.qk_rope_head_dim // 2,
        dtype=torch.complex64,
    )
    position_ids = torch.arange(seqlen).unsqueeze(0).expand(bsz, -1)

    injected = ops(
        o,
        freqs_cis,
        bsz,
        seqlen,
        config.o_groups,
        wrapped.wo_a,
        wrapped.wo_b,
        position_ids,
    )

    # The absorption must consume the adapter exactly like a merged weight.
    reference = ops(
        o,
        freqs_cis,
        bsz,
        seqlen,
        config.o_groups,
        _merged_linear(wrapped.wo_a),
        _merged_linear(wrapped.wo_b),
        position_ids,
    )
    torch.testing.assert_close(injected.detach(), reference.detach())

    # Regression guard: before the fix the absorption read ``wo_a.weight``
    # (base only), so the wo_a adapter stayed out of the autograd graph.
    injected.sum().backward()
    grad_a = wrapped.wo_a.lora_A["default"].weight.grad
    grad_b = wrapped.wo_a.lora_B["default"].weight.grad
    assert grad_a is not None and grad_a.abs().sum() > 0
    assert grad_b is not None and grad_b.abs().sum() > 0

    # Merging folds the delta into the base weight; the absorption output
    # must stay identical after the merge.
    lora_holder.merge_adapter()
    merged_out = ops(
        o,
        freqs_cis,
        bsz,
        seqlen,
        config.o_groups,
        wrapped.wo_a,
        wrapped.wo_b,
        position_ids,
    )
    torch.testing.assert_close(merged_out.detach(), reference.detach())
