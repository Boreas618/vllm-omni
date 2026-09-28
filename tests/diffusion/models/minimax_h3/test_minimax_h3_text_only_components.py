# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Text-only deployments without VAE encoders or the Qwen3-VL vision tower."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from tests.diffusion.models.minimax_h3.test_minimax_h3_contract import _write_partition_index
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import PreparedEncoderInputs

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_TEXT_HIDDEN = 32


@pytest.fixture
def tiny_qwen3vl(tmp_path, monkeypatch):
    """A tiny Qwen3-VL config directory and its Hugging Face state dict."""
    import vllm.model_executor.parameter as vllm_parameter
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from vllm_omni.diffusion.models.minimax_h3 import encoder as encoder_module

    monkeypatch.setattr(vllm_parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vllm_parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(encoder_module, "MINIMAX_H3_QWEN3VL_HIDDEN_DIM", _TEXT_HIDDEN)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=160_000,
            hidden_size=_TEXT_HIDDEN,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True},
        ),
        vision_config=dict(
            depth=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=_TEXT_HIDDEN,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=2,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0],
        ),
    )
    torch.manual_seed(0)
    state_dict = Qwen3VLForConditionalGeneration(config).state_dict()
    config.save_pretrained(tmp_path / "text_encoder")
    return str(tmp_path / "text_encoder"), state_dict


def _encoder(path: str, *, load_vision: bool):
    from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _SingleRankEncoderGroup

    return MiniMaxH3Qwen3VLEncoder(
        path,
        device=torch.device("cpu"),
        load_model=True,
        encoder_group=_SingleRankEncoderGroup(0),
        load_vision=load_vision,
    )


def test_encoder_without_vision_tower_loads_and_encodes_text(tiny_qwen3vl, caplog):
    path, state_dict = tiny_qwen3vl
    full = _encoder(path, load_vision=True)
    text_only = _encoder(path, load_vision=False)

    full_loaded = full.load_weights(state_dict.items())
    with caplog.at_level(logging.WARNING):
        text_loaded = text_only.load_weights(state_dict.items())

    assert any(name.startswith("model.visual.") for name in state_dict)
    assert not any(name.startswith("vision.") for name, _ in text_only.named_parameters())
    assert text_loaded == {name for name in full_loaded if not name.startswith("vision.")}
    assert "no target parameter" not in caplog.text

    ids = torch.tensor([1, 2, 3, 4, 5])
    assert torch.equal(text_only.encode_ids(ids), full.encode_ids(ids))
    with pytest.raises(ValueError, match="vision tower"):
        text_only.encode_ids(ids, pixel_values=torch.zeros(4, 96), image_grid_thw=torch.tensor([[1, 2, 2]]))


def test_encoder_without_vision_tower_keeps_strict_text_loading(tiny_qwen3vl):
    path, state_dict = tiny_qwen3vl
    partial = [(name, tensor) for name, tensor in state_dict.items() if "layers.1.mlp.down_proj" not in name]

    with pytest.raises(RuntimeError, match="weights not loaded"):
        _encoder(path, load_vision=False).load_weights(partial)


def _pipeline(monkeypatch, tmp_path, model_loaded: dict[str, bool]):
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    _write_partition_index(tmp_path / "FL2VA", partition="fl2va", tasks=["t2va", "fl2va"])
    for component in ("transformer", "video_vae", "audio_vae"):
        (tmp_path / "FL2VA" / component).mkdir(parents=True)
    options: dict[str, dict[str, Any]] = {}

    def component(name):
        def create(path, *args, **kwargs):
            options[name] = kwargs
            return torch.nn.Module()

        return create

    monkeypatch.setattr(pipeline_module, "MiniMaxH3DiTModel", lambda *args, **kwargs: torch.nn.Module())
    monkeypatch.setattr(pipeline_module, "MiniMaxH3Qwen3VLEncoder", component("text_encoder"))
    monkeypatch.setattr(pipeline_module, "MiniMaxH3VideoVAE", component("video_vae"))
    monkeypatch.setattr(pipeline_module, "MiniMaxH3AudioVAE", component("audio_vae"))
    monkeypatch.setattr(pipeline_module.Qwen2TokenizerFast, "from_pretrained", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_module.Qwen3VLProcessor, "from_pretrained", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_module, "download_weights_from_hf_specific", lambda **kwargs: str(tmp_path))
    od_config = OmniDiffusionConfig(
        model="MiniMaxAI/MiniMax-H3",
        task_type="fl2va",
        parallel_config=DiffusionParallelConfig(cfg_parallel_size=1, text_encoder_tp_size=1),
    )
    od_config.model_loaded = model_loaded
    return pipeline_module.MiniMaxH3Pipeline(od_config=od_config), options


def test_local_text_encoder_serves_without_media_encoders(monkeypatch, tmp_path):
    pipeline, options = _pipeline(monkeypatch, tmp_path, {"vae_encoder": False, "vision_tower": False})

    assert pipeline.load_text_encoder and pipeline.text_encoder is not None
    assert options["text_encoder"]["load_vision"] is False
    assert options["video_vae"]["decode_only"] is True
    assert options["audio_vae"]["decode_only"] is True
    assert pipeline._offload_plan.encoder_block_attrs["text_encoder"] == ("text_model.layers",)

    default, default_options = _pipeline(monkeypatch, tmp_path / "default", {})
    assert default_options["text_encoder"]["load_vision"] is True
    assert default_options["video_vae"]["decode_only"] is False
    assert default._offload_plan.encoder_block_attrs["text_encoder"] == ("vision.blocks", "text_model.layers")


def _media(**updates) -> MiniMaxH3EncoderMediaInput:
    return MiniMaxH3EncoderMediaInput(
        task="t2va", height=64, width=96, num_frames=22, latent_t=7, audio_t=37, **updates
    )


def _prepared(*, media=None, images=(), qwen_videos=()) -> PreparedEncoderInputs:
    return PreparedEncoderInputs(
        prompt="p",
        media=media or _media(),
        images=list(images),
        qwen_videos=list(qwen_videos),
        video_timestamps=[],
        condition_labels=[],
    )


_WAVEFORM = (torch.zeros(2, 320), 32_000)


@pytest.mark.parametrize(
    "prepared,needs_vae,needs_vision",
    [
        (_prepared(), False, False),
        (_prepared(media=_media(images=(torch.zeros(3, 64, 96),)), images=[Image.new("RGB", (96, 64))]), True, True),
        (_prepared(media=_media(videos=(torch.zeros(5, 64, 96, 3),)), qwen_videos=[(None, {})]), True, True),
        (_prepared(media=_media(audios=(_WAVEFORM,), audio_mode="lock_source")), True, False),
        (_prepared(media=_media(video_audios=(_WAVEFORM,))), True, False),
        (_prepared(media=_media(video_edit=torch.zeros(22, 64, 96, 3))), True, False),
        (_prepared(media=_media(audio_edit=_WAVEFORM)), True, False),
    ],
    ids=["text", "image", "video", "lock-source-audio", "video-soundtrack", "video-edit", "audio-edit"],
)
@pytest.mark.parametrize("vae_encoder,vision_tower", [(False, True), (True, False), (False, False)])
def test_requests_needing_missing_encoders_are_rejected(prepared, needs_vae, needs_vision, vae_encoder, vision_tower):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = SimpleNamespace(load_text_encoder=True, load_vae_encoder=vae_encoder, load_vision_tower=vision_tower)
    rejected = (needs_vae and not vae_encoder) or (needs_vision and not vision_tower)
    if rejected:
        with pytest.raises(OmniClientError, match="this MiniMax H3 deployment has no"):
            MiniMaxH3Pipeline._check_local_media_encoders(pipeline, prepared)
    else:
        MiniMaxH3Pipeline._check_local_media_encoders(pipeline, prepared)
