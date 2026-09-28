# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online AdaLN: projections built from the weight stream replace resident weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.diffusion.cache.test_teacache_extractors import (
    _minimax_h3_sample_kwargs,
    _MiniMaxH3FakeAttention,
    _MiniMaxH3FakeLinear,
    _MiniMaxH3FakeMergedLinear,
    _MiniMaxH3FakeQKVLinear,
)
from vllm_omni.diffusion.cache.teacache.extractors import extract_minimax_h3_context
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3
from vllm_omni.diffusion.models.minimax_h3.adaln_cache import schedule_contract, timestep_plans
from vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint import (
    FASTH3_V2_BASE_SCHEDULE,
    FastH3CheckpointSpec,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Eleven blocks make the checkpoint's lexicographic key order differ from the
# numeric block order (transformer_blocks.10 sorts before transformer_blocks.2).
_LAYERS = 11
_HIDDEN = 8
_T_DIM = 4
_PLANS = timestep_plans(
    schedule_contract(
        mode="t2va",
        num_steps=FASTH3_V2_BASE_SCHEDULE.num_inference_steps,
        base_schedule=FASTH3_V2_BASE_SCHEDULE.base_schedule,
        flow_shift=10.0,
        audio_flow_shift=3.0,
    )
)


@pytest.fixture(autouse=True)
def cpu_linears(monkeypatch):
    import vllm.distributed

    # The resident runtime projection cache votes across the TP group.
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(h3, "ColumnParallelLinear", _MiniMaxH3FakeLinear)
    monkeypatch.setattr(h3, "MergedColumnParallelLinear", _MiniMaxH3FakeMergedLinear)
    monkeypatch.setattr(h3, "QKVParallelLinear", _MiniMaxH3FakeQKVLinear)
    monkeypatch.setattr(h3, "RowParallelLinear", _MiniMaxH3FakeLinear)
    monkeypatch.setattr(h3, "Attention", _MiniMaxH3FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)


def _od_config(**updates):
    arch = {
        "num_layers": _LAYERS,
        "token_refiner_num_layers": 1,
        "hidden_size": _HIDDEN,
        "num_attention_heads": 1,
        "attention_head_dim": _HIDDEN,
        "ffn_hidden_size": 16,
        "latents_dim": 2,
        "audio_latents_dim": 2,
        "patch_size": (1, 2, 2),
        "text_dim": 6,
        "timestep_input_dim": 4,
        "time_embed_hidden_size": 8,
        "time_embed_dim": _T_DIM,
        "adaln_out_features": 18 * _HIDDEN,
        "final_adaln_out_features": 2 * _HIDDEN,
        "rope_inv_freq_len": 1,
    }
    values = dict(
        tf_model_config=arch,
        parallel_config=SimpleNamespace(ulysses_degree=1),
        cache_config={},
        enforce_eager=True,
        diffusion_compile_granularity="regional",
        diffusion_compile_dynamic=True,
    )
    return SimpleNamespace(**(values | updates))


def _model(*, online: bool, **config_updates) -> h3.MiniMaxH3DiTModel:
    model = h3.MiniMaxH3DiTModel(
        _od_config(**config_updates),
        diffusers_weights=True,
        adaln_plans=_PLANS if online else None,
    )
    # Checkpoint buffer in the native layout; one frequency per axis.
    model.rope.inv_freq.fill_(1.0)
    for module in model.modules():
        if isinstance(module, h3.MiniMaxH3Attention):
            module.rope._forward_method = module.rope.forward_native
    return model.eval()


def _checkpoint(seed: int = 0) -> dict[str, torch.Tensor]:
    """Diffusers-named time embedder, AdaLN and norm tensors, stored as BF16."""
    generator = torch.Generator().manual_seed(seed)

    def tensor(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator).to(torch.bfloat16)

    weights = {
        "time_embedder.linear_1.weight": tensor(8, 4),
        "time_embedder.linear_1.bias": tensor(8),
        "time_embedder.linear_2.weight": tensor(_T_DIM, 8),
        "time_embedder.linear_2.bias": tensor(_T_DIM),
        "norm_out.linear.weight": tensor(2 * _HIDDEN, _T_DIM),
        "norm_out.linear.bias": tensor(2 * _HIDDEN),
        "norm_out.norm.weight": tensor(_HIDDEN),
    }
    for index in range(_LAYERS):
        prefix = f"transformer_blocks.{index}"
        weights[f"{prefix}.adaln_proj.linear.weight"] = tensor(18 * _HIDDEN, _T_DIM)
        weights[f"{prefix}.adaln_proj.linear.bias"] = tensor(18 * _HIDDEN)
        weights[f"{prefix}.norm1.weight"] = tensor(_HIDDEN)
    return weights


def _stream(weights: dict[str, torch.Tensor], order: str) -> list[tuple[str, torch.Tensor]]:
    # A checkpoint shard lists its keys sorted: norm_out.linear precedes the
    # time embedder, and the blocks follow it.
    names = sorted(weights)
    if order == "adaln_before_time_embedder":
        # Multithreaded loading yields whole shards in completion order, so a
        # later shard's blocks can arrive before the time embedder.
        late = [name for name in names if name.startswith(("transformer_blocks.10.", "transformer_blocks.7."))]
        names = late + [name for name in names if name not in late]
    elif order == "split_pairs":
        # A projection's bias and weight can sit in different shards.
        names = [name for name in names if name.endswith(".bias")] + [
            name for name in names if not name.endswith(".bias")
        ]
    return [(name, weights[name]) for name in names]


def _expected_projection(model, weights, prefix, plan) -> torch.Tensor:
    with torch.inference_mode():
        embedding = model.time_embedder(plan)
    return nn.functional.linear(
        nn.functional.silu(embedding).to(torch.bfloat16),
        weights[f"{prefix}.weight"],
        weights[f"{prefix}.bias"],
    )


@pytest.mark.parametrize("order", ["checkpoint", "adaln_before_time_embedder", "split_pairs"])
def test_table_equals_per_layer_projection(order):
    weights = _checkpoint()
    model = _model(online=True)

    loaded = model.load_weights(_stream(weights, order))

    table = model.adaln_table
    assert table.ready and model._adaln_builder is None
    assert not any("adaln_proj.linear" in name for name, _ in model.named_parameters())
    assert not any("adaln_proj.linear" in name for name in loaded)
    assert {"time_embedder.proj_in.weight", "final_layer.norm.weight", "blocks.10.norm1.weight"} <= loaded
    assert tuple(table.block_rows.shape) == (_LAYERS, len(_PLANS), 2, 18 * _HIDDEN)
    for plan_index, plan in enumerate(_PLANS):
        count = plan.numel()
        for layer in range(_LAYERS):
            expected = _expected_projection(model, weights, f"transformer_blocks.{layer}.adaln_proj.linear", plan)
            assert torch.equal(table.block_rows[layer, plan_index, :count], expected)
        expected = _expected_projection(model, weights, "norm_out.linear", plan)
        assert torch.equal(table.final_rows[plan_index, :count], expected)


@pytest.mark.parametrize(
    "change,match",
    [
        ("drop_block", "missing checkpoint projections"),
        ("drop_time_embedder", "time embedder"),
        ("duplicate", "duplicate AdaLN weight"),
    ],
)
def test_incomplete_or_repeated_adaln_stream_is_rejected(change, match):
    stream = _stream(_checkpoint(), "checkpoint")
    if change == "drop_block":
        stream = [item for item in stream if item[0] != "transformer_blocks.4.adaln_proj.linear.weight"]
    elif change == "drop_time_embedder":
        stream = [item for item in stream if not item[0].startswith("time_embedder.linear_2.")]
    else:
        # A native-named repeat bypasses the Diffusers duplicate-name check.
        model = _model(online=True)
        model._diffusers_weights = False
        native = [("blocks.3.adaln_proj.linear.bias", torch.zeros(18 * _HIDDEN, dtype=torch.bfloat16))] * 2
        with pytest.raises(ValueError, match=match):
            model.load_weights(native)
        return
    with pytest.raises(ValueError, match=match):
        _model(online=True).load_weights(stream)


def _forward_kwargs(plan: torch.Tensor) -> dict:
    kwargs = _minimax_h3_sample_kwargs(seq_len=8)
    inverse = torch.zeros(8, dtype=torch.long)
    # Video, text and padding rows use the video timestep; audio rows use the
    # audio timestep, which is the larger value of every FastH3 plan.
    inverse[kwargs["audio_pos_info"]["position_ids"]] = plan.numel() - 1
    return kwargs | {"unique_timesteps": plan.clone(), "inverse_indices": inverse}


def _resident_and_online():
    torch.manual_seed(0)
    resident = _model(online=False)
    online = _model(online=True)
    shared = {name: value for name, value in resident.state_dict().items() if "adaln_proj" not in name}
    missing, unexpected = online.load_state_dict(shared, strict=False)
    assert not unexpected and all(name.startswith("adaln_table.") for name in missing)
    stream = _stream(_checkpoint(seed=1), "checkpoint")
    resident.load_weights(stream)
    online.load_weights(stream)
    return resident, online


def test_online_forward_matches_resident_forward():
    resident, online = _resident_and_online()
    for plan in (_PLANS[0], _PLANS[-1]):
        kwargs = _forward_kwargs(plan)
        with torch.inference_mode():
            expected = resident(**kwargs)
            actual = online(**kwargs)
        for reference, value in zip(expected, actual, strict=True):
            assert torch.equal(reference, value)


def test_teacache_extractor_uses_online_projections():
    resident, online = _resident_and_online()
    kwargs = _forward_kwargs(_PLANS[2])
    with torch.inference_mode():
        expected = resident(**kwargs)
        context = extract_minimax_h3_context(online, **kwargs)
        actual = context.postprocess(*context.run_transformer_blocks())
    for reference, value in zip(expected, actual, strict=True):
        assert torch.equal(reference, value)


@pytest.mark.parametrize(
    "timesteps",
    [
        torch.tensor([0.5]),
        # Two co-batched step-mode requests at different steps.
        torch.cat((_PLANS[0], _PLANS[1])).unique(sorted=True),
        # audio_mode=lock_source pins audio rows at t=1.
        torch.tensor([_PLANS[3][0].item(), 1.0]),
    ],
    ids=["other-schedule", "step-union", "lock-source"],
)
def test_timesteps_outside_table_are_rejected(timesteps):
    online = _model(online=True)
    online.load_weights(_stream(_checkpoint(), "checkpoint"))
    with pytest.raises(ValueError, match="no projections"):
        online(**_forward_kwargs(timesteps.to(torch.float32)))


def test_unloaded_table_is_rejected():
    with pytest.raises(RuntimeError, match="incomplete"):
        _model(online=True).adaln_modulation(_PLANS[0])


@pytest.mark.parametrize(
    "setup,match",
    [
        ("tensor_parallel", "tensor_parallel_size=1"),
        ("quantized", "unquantized"),
        ("full_compile", "full-model compile"),
    ],
)
def test_transformer_rejects_unsupported_online_configurations(monkeypatch, setup, match):
    config = _od_config()
    quant_config = None
    if setup == "tensor_parallel":
        monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 2)
    elif setup == "quantized":
        quant_config = object()
    else:
        config.diffusion_compile_granularity = "full"
    with pytest.raises(ValueError, match=match):
        h3.MiniMaxH3DiTModel(config, quant_config, diffusers_weights=True, adaln_plans=_PLANS)


def _pipeline(checkpoint: bool = True) -> MiniMaxH3Pipeline:
    pipeline = object.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._fasth3_checkpoint = FastH3CheckpointSpec() if checkpoint else None
    pipeline._base_schedule_by_partition = {"fl2va": FASTH3_V2_BASE_SCHEDULE}
    pipeline.default_video_shift = 10.0
    pipeline.default_audio_shift = 3.0
    return pipeline


def _deployment(cache: dict | None = None, **updates) -> SimpleNamespace:
    values = dict(
        cache_config=DiffusionCacheConfig.from_dict({"minimax_h3_adaln_online": True} | (cache or {})),
        cache_backend="none",
        parallel_config=SimpleNamespace(use_hsdp=False),
        host_weight_runtime_mode="disabled",
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
    )
    return SimpleNamespace(**(values | updates))


def test_pipeline_resolves_the_pinned_fasth3_plans():
    plans = _pipeline()._online_adaln_plans(_deployment(), "fl2va")
    assert len(plans) == FASTH3_V2_BASE_SCHEDULE.num_inference_steps
    assert all(torch.equal(plan, expected) for plan, expected in zip(plans, _PLANS, strict=True))
    default = _deployment(cache={"minimax_h3_adaln_online": False})
    assert _pipeline()._online_adaln_plans(default, "fl2va") is None


@pytest.mark.parametrize(
    "pipeline_checkpoint,deployment,match",
    [
        (False, {}, "FastH3 V2 checkpoint"),
        (True, {"cache": {"minimax_h3_adaln_online": "yes"}}, "must be a boolean"),
        (True, {"cache": {"minimax_h3_adaln_cache_path": "/tmp/sidecar"}}, "sidecar"),
        (True, {"cache_backend": "cache_dit"}, "Cache-DiT"),
        (True, {"enable_cpu_offload": True}, "offload"),
        (True, {"enable_layerwise_offload": True}, "offload"),
        (True, {"enable_distributed_layerwise_offload": True}, "offload"),
        (True, {"parallel_config": SimpleNamespace(use_hsdp=True)}, "HSDP"),
        (True, {"host_weight_runtime_mode": "preferred"}, "host_weight_runtime_mode"),
    ],
)
def test_pipeline_rejects_unsupported_online_deployments(pipeline_checkpoint, deployment, match):
    with pytest.raises(ValueError, match=match):
        _pipeline(pipeline_checkpoint)._online_adaln_plans(_deployment(**deployment), "fl2va")
