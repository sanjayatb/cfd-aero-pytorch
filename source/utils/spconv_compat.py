"""Compatibility layer for spconv.

If the real spconv package is unavailable (common on CPU-only or minimal setups),
we provide light-weight fallbacks that mimic the subset of APIs used by the
PointTransformerV3 implementation. These fallbacks operate on dense tensors and
should not be considered drop-in replacements for high-performance sparse ops,
but they allow the model to execute for experimentation and debugging.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn as nn

try:  # pragma: no cover - defer to the real implementation when available
    import spconv.pytorch as spconv  # type: ignore
except ImportError:  # pragma: no cover - exercised when spconv is missing

    @dataclass
    class _SparseMeta:
        indices: torch.Tensor
        spatial_shape: Optional[Sequence[int]]
        batch_size: Optional[int]

    class SparseConvTensor:
        """Minimal stand-in for spconv.SparseConvTensor."""

        def __init__(
            self,
            features: torch.Tensor,
            indices: Optional[torch.Tensor] = None,
            spatial_shape: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None,
        ):
            self.features = features
            device = features.device
            if indices is None:
                indices = torch.zeros(
                    (features.shape[0], 4), dtype=torch.long, device=device
                )
            self._meta = _SparseMeta(
                indices=indices.to(device),
                spatial_shape=spatial_shape,
                batch_size=batch_size,
            )

        @property
        def indices(self) -> torch.Tensor:
            return self._meta.indices

        @property
        def spatial_shape(self) -> Optional[Sequence[int]]:
            return self._meta.spatial_shape

        @property
        def batch_size(self) -> Optional[int]:
            return self._meta.batch_size

        def replace_feature(self, features: torch.Tensor) -> "SparseConvTensor":
            return SparseConvTensor(
                features=features,
                indices=self._meta.indices,
                spatial_shape=self._meta.spatial_shape,
                batch_size=self._meta.batch_size,
            )


    class _SparseModule(nn.Module):
        """Base class to allow isinstance checks via modules.is_spconv_module."""


    class SubMConv3d(_SparseModule):
        """Dense linear projection that emulates SubmanifoldConv behaviour."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Any,
            stride: Any = 1,
            padding: Any = 0,
            bias: bool = True,
            indice_key: Optional[str] = None,
            **_: Any,
        ):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels, bias=bias)

        def forward(self, input: Any) -> Any:
            if isinstance(input, SparseConvTensor):
                out = self.linear(input.features)
                return input.replace_feature(out)
            return self.linear(input)


    class _ModulesNamespace:
        @staticmethod
        def is_spconv_module(module: nn.Module) -> bool:
            return isinstance(module, _SparseModule)


    spconv = types.SimpleNamespace(  # type: ignore
        SparseConvTensor=SparseConvTensor,
        SubMConv3d=SubMConv3d,
        modules=_ModulesNamespace(),
    )

