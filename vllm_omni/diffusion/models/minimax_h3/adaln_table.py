# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online AdaLN: pinned-schedule projections built from the weight stream.

A FastH3 checkpoint denoises every plain T2VA request on the same timestep
plans, so the output of each AdaLN projection is fixed per (layer, plan). The
online table computes those outputs once while the checkpoint streams in and
never materializes the projection weights on the device (24 GiB for the full
DiT). Requests whose forward timesteps fall outside the table are rejected;
they need the resident projection weights.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .adaln_cache import projection_names

if TYPE_CHECKING:
    from .minimax_h3_transformer import MiniMaxH3AdalnProj

_FINAL_PROJECTION = "final_layer.adaln_proj.linear"


def plan_key(timesteps: torch.Tensor) -> tuple[int, ...]:
    """Identify a timestep plan by the exact FP32 bits of its sorted values.

    The time embedder consumes FP32 timesteps, so equal bits are exactly the
    inputs that reproduce a stored projection. Reading a device tensor costs
    one small device-to-host copy.
    """
    values = timesteps.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
    return tuple(values.view(torch.int32).tolist())


def numerical_settings(device_type: str) -> tuple[Any, ...]:
    """Global matmul and autocast state that selects the projection kernels."""
    return (
        torch.get_float32_matmul_precision(),
        torch.is_autocast_enabled(device_type),
        torch.get_autocast_dtype(device_type),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    )


class MiniMaxH3AdalnTable(nn.Module):
    """AdaLN projections of every DiT block and the final layer per timestep plan.

    ``block_rows`` is ``[num_layers, num_plans, width, block_features]`` and
    ``final_rows`` is ``[num_plans, width, final_features]``, both BF16, where
    ``width`` is the longest plan. For layer ``i`` and plan ``p`` of length
    ``M``, ``block_rows[i, p, :M]`` is the contiguous ``[M, 18H]`` output a
    resident block computes from ``time_embedder(plan)``; rows past ``M`` are
    padding and never read. Both tables are non-persistent buffers: they
    follow device moves and sleep-mode buffer restore, but are not checkpoint
    state.
    """

    block_rows: torch.Tensor
    final_rows: torch.Tensor

    def __init__(
        self,
        plans: Sequence[torch.Tensor],
        *,
        num_layers: int,
        block_features: int,
        final_features: int,
    ) -> None:
        super().__init__()
        if not plans:
            raise ValueError("online AdaLN needs at least one timestep plan")
        self.plans: tuple[torch.Tensor, ...] = tuple(plan.detach().to("cpu", torch.float32).clone() for plan in plans)
        self._plan_index: dict[tuple[int, ...], int] = {}
        for index, plan in enumerate(self.plans):
            if plan.dim() != 1 or not plan.numel() or not torch.equal(plan, plan.unique(sorted=True)):
                raise ValueError(f"online AdaLN plans must be sorted unique 1-D timesteps, got {plan.tolist()}")
            if self._plan_index.setdefault(plan_key(plan), index) != index:
                raise ValueError(f"duplicate online AdaLN plan {plan.tolist()}")
        self.plan_lengths = tuple(int(plan.numel()) for plan in self.plans)
        width = max(self.plan_lengths)
        self.register_buffer(
            "block_rows",
            torch.zeros(num_layers, len(self.plans), width, block_features, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "final_rows",
            torch.zeros(len(self.plans), width, final_features, dtype=torch.bfloat16),
            persistent=False,
        )
        # Set by ``mark_ready`` once every projection has been written.
        self.numerics: tuple[Any, ...] | None = None

    @property
    def ready(self) -> bool:
        return self.numerics is not None

    def mark_ready(self, numerics: tuple[Any, ...]) -> None:
        self.numerics = numerics

    def lookup(self, unique_timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return this forward's ``(block rows [L, M, 18H], final rows [M, 2H])``.

        ``unique_timesteps`` is the forward's sorted unique timestep vector.
        Raises ``ValueError`` when no stored plan has identical FP32 bits:
        latent-mask edits, ``audio_mode=lock_source``, co-batched step-mode
        requests at different steps, and any schedule other than the pinned
        one produce such vectors, and the table has no weights to compute them.
        """
        if not self.ready:
            raise RuntimeError("online AdaLN table is incomplete; the checkpoint AdaLN weights were not loaded")
        index = self._plan_index.get(plan_key(unique_timesteps))
        if index is None:
            values = unique_timesteps.detach().reshape(-1).cpu().tolist()
            raise ValueError(
                f"online AdaLN has no projections for unique timesteps {values}; it covers only the "
                f"{len(self.plans)} plans of the pinned FastH3 T2VA schedule. Latent-mask edits, "
                "audio_mode=lock_source, co-batched step-mode requests and other schedules need the "
                "resident AdaLN weights (remove minimax_h3_adaln_online)."
            )
        device_type = self.block_rows.device.type
        if numerical_settings(device_type) != self.numerics:
            raise RuntimeError(
                "online AdaLN projections were built under different matmul or autocast settings "
                f"({self.numerics}) than this forward ({numerical_settings(device_type)})"
            )
        count = self.plan_lengths[index]
        return self.block_rows[:, index, :count], self.final_rows[index, :count]


class MiniMaxH3AdalnTableBuilder:
    """Fill a :class:`MiniMaxH3AdalnTable` from the AdaLN tensors of the weight stream.

    Every projection runs through a carrier :class:`MiniMaxH3AdalnProj` built
    exactly like a resident one (same ``ColumnParallelLinear``, dtype and
    linear backend), loaded through the parameter's own ``weight_loader``, and
    evaluated once per plan at that plan's row count ``M`` because GEMM kernel
    selection depends on ``M``. Block projections are traced by
    ``torch.compile`` with the regional-compile settings when the served
    blocks are compiled, since a resident block computes
    ``silu -> bf16 -> linear`` inside its compiled region; the final layer is
    never regionally compiled and projects eagerly.

    Multithreaded loading yields checkpoint shards in completion order, so
    AdaLN tensors may precede the time embedder. They are held as host
    references until :meth:`set_embeddings` supplies the plan embeddings.
    """

    def __init__(
        self,
        table: MiniMaxH3AdalnTable,
        *,
        num_layers: int,
        block_projection: MiniMaxH3AdalnProj,
        final_projection: MiniMaxH3AdalnProj,
        compile_blocks: bool,
        compile_dynamic: bool,
    ) -> None:
        self.table = table
        self.names = projection_names(num_layers)
        self._block_projection = block_projection
        self._final_projection = final_projection
        self._project_block: Callable[[torch.Tensor], torch.Tensor] = block_projection.project
        if compile_blocks:
            # The carrier's parameters stay the same objects while each layer's
            # weights are loaded into them in place, so one compiled graph
            # serves all layers without recompilation.
            def project_block(embedding: torch.Tensor) -> torch.Tensor:
                return block_projection.project(embedding)

            self._project_block = torch.compile(project_block, dynamic=compile_dynamic)
        self._pending: dict[str, dict[str, torch.Tensor]] = {}
        self._projected: set[str] = set()
        self._embeddings: tuple[torch.Tensor, ...] | None = None
        self._numerics: tuple[Any, ...] | None = None

    @property
    def awaiting_embeddings(self) -> bool:
        return self._embeddings is None

    def add(self, name: str, tensor: torch.Tensor) -> None:
        """Consume one AdaLN tensor (native name) from the weight stream."""
        prefix, _, kind = name.rpartition(".")
        pending = self._pending.setdefault(prefix, {})
        if prefix in self._projected or kind in pending:
            raise ValueError(f"duplicate AdaLN weight for online AdaLN: {name}")
        pending[kind] = tensor
        if self._embeddings is not None and len(pending) == 2:
            self._project(prefix, self._pending.pop(prefix))

    def set_embeddings(self, embeddings: Iterable[torch.Tensor]) -> None:
        """Supply ``time_embedder(plan)`` for every plan and flush held tensors."""
        self._embeddings = tuple(embeddings)
        device_type = self._embeddings[0].device.type
        self._numerics = numerical_settings(device_type)
        for prefix in [prefix for prefix, pending in self._pending.items() if len(pending) == 2]:
            self._project(prefix, self._pending.pop(prefix))

    def _project(self, prefix: str, tensors: dict[str, torch.Tensor]) -> None:
        final = prefix == _FINAL_PROJECTION
        projection = self._final_projection if final else self._block_projection
        linear = projection.linear
        for kind in ("weight", "bias"):
            param = getattr(linear, kind)
            getattr(param, "weight_loader", default_weight_loader)(param, tensors[kind])
        project = projection.project if final else self._project_block
        rows = self.table.final_rows if final else self.table.block_rows[int(prefix.split(".")[1])]
        assert self._embeddings is not None
        # Match the served forward, which runs the DiT under inference mode.
        with torch.inference_mode():
            for index, embedding in enumerate(self._embeddings):
                rows[index, : embedding.shape[0]].copy_(project(embedding))
        self._projected.add(prefix)

    def finish(self) -> None:
        """Mark the table ready, or raise if the stream missed any projection."""
        expected = {name.rpartition(".")[0] for name in self.names}
        missing = sorted((expected - self._projected) | set(self._pending))
        if self._embeddings is None:
            raise ValueError("online AdaLN requires the time embedder weights in the checkpoint")
        if missing:
            raise ValueError(f"online AdaLN is missing checkpoint projections: {missing}")
        assert self._numerics is not None
        self.table.mark_ready(self._numerics)


__all__ = [
    "MiniMaxH3AdalnTable",
    "MiniMaxH3AdalnTableBuilder",
    "numerical_settings",
    "plan_key",
]
