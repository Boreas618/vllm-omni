# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout, VideoTokenSpan
from vllm_omni.diffusion.models.minimax_h3.attention.fastvideo_h3 import MiniMaxH3VSAImpl

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cuda
@pytest.mark.parametrize("native", ["0", "1"])
@pytest.mark.parametrize(("video_tiles", "selected_tiles"), [(5, 1), (6, 2), (10, 2)])
def test_trained_sparsity_preserves_prefix_and_selects_video_fraction(monkeypatch, native, video_tiles, selected_tiles):
    monkeypatch.setenv("FASTVIDEO_VSA_SM100A", native)
    # Each temporal slab is one complete 4x4x4 tile. Increasing keys give an
    # unambiguous ranking, so the reference mask follows the trained 80% policy
    # without depending on the implementation's routing or tiling helpers.
    length = 1 + 64 * video_tiles
    query = torch.ones(1, length, 1, 128, device="cuda", dtype=torch.bfloat16)
    block_values = torch.arange(video_tiles + 1, device="cuda").repeat_interleave(64)[63:]
    key = (block_values / (video_tiles * 128**0.5)).to(query.dtype)[None, :, None, None].expand_as(query).contiguous()
    value = (block_values / video_tiles).to(query.dtype)[None, :, None, None].expand_as(query).contiguous()
    metadata = AttentionMetadata(
        extra={"vsa_h3_prefix_segments": (1,), "vsa_h3_sparsity": 0.8},
        video_layout=VideoTokenLayout(
            used_len=length,
            video_spans=(VideoTokenSpan(start=1, latent_grid=(4 * video_tiles, 4, 4), role="target"),),
        ),
    )
    impl = MiniMaxH3VSAImpl(num_heads=1, head_size=128, softmax_scale=128**-0.5)
    output = impl.forward_cuda(query, key, value, metadata)

    mask = torch.zeros(length, length, device="cuda", dtype=torch.bool)
    mask[0] = True
    mask[:, 0] = True
    mask[:, -selected_tiles * 64 :] = True
    reference = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=mask
    ).transpose(1, 2)
    torch.testing.assert_close(output, reference)


@pytest.mark.cpu
@pytest.mark.parametrize("invalid_layout", [False, True])
def test_trained_sparsity_rejects_missing_or_inconsistent_geometry(invalid_layout):
    query = torch.ones(1, 65, 1, 128, dtype=torch.bfloat16)
    metadata = AttentionMetadata(extra={"vsa_h3_sparsity": 0.8})
    if invalid_layout:
        metadata.extra["vsa_h3_prefix_segments"] = (2,)
        metadata.video_layout = VideoTokenLayout(
            used_len=65,
            video_spans=(VideoTokenSpan(start=1, latent_grid=(4, 4, 4), role="target"),),
        )
    impl = MiniMaxH3VSAImpl(num_heads=1, head_size=128, softmax_scale=128**-0.5)
    with pytest.raises(ValueError, match="layout metadata|prefix segments"):
        impl.forward_cuda(query, query, query, metadata)
