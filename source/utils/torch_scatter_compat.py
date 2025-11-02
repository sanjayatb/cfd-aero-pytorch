"""Compatibility helpers for torch_scatter functions used in PointTransformerV3."""

from __future__ import annotations

from typing import Literal

import torch

Reduce = Literal["sum", "mean", "max", "min"]

try:  # pragma: no cover - prefer the optimised implementation
    from torch_scatter import segment_csr as _segment_csr  # type: ignore

    def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: Reduce = "sum"):
        return _segment_csr(src, indptr, reduce=reduce)

except ImportError:  # pragma: no cover - exercised when torch_scatter missing

    def _as_segment_ids(indptr: torch.Tensor) -> torch.Tensor:
        """Expand a CSR pointer array into explicit segment ids."""
        counts = indptr[1:] - indptr[:-1]
        return torch.repeat_interleave(
            torch.arange(len(counts), device=indptr.device, dtype=torch.long),
            counts,
        )

    def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: Reduce = "sum"):
        """Pure PyTorch fallback for torch_scatter.segment_csr."""

        if src.numel() == 0:
            return torch.zeros(
                (indptr.numel() - 1,) + src.shape[1:], device=src.device, dtype=src.dtype
            )

        segment_ids = _as_segment_ids(indptr)
        output_shape = (indptr.numel() - 1,) + src.shape[1:]

        if reduce in ("sum", "mean"):
            out = torch.zeros(output_shape, device=src.device, dtype=src.dtype)
            out.scatter_add_(
                0, segment_ids.view(-1, *([1] * (src.dim() - 1))).expand_as(src), src
            )
            if reduce == "mean":
                counts = (indptr[1:] - indptr[:-1]).clamp_min(1).view(-1, *([1] * (src.dim() - 1)))
                out = out / counts
            return out

        if reduce == "max":
            out = torch.full(
                output_shape,
                torch.finfo(src.dtype).min if src.is_floating_point() else torch.iinfo(src.dtype).min,
                device=src.device,
                dtype=src.dtype,
            )
            out.scatter_reduce_(
                0,
                segment_ids.view(-1, *([1] * (src.dim() - 1))).expand_as(src),
                src,
                reduce="amax",
                include_self=False,
            )
            return out

        if reduce == "min":
            out = torch.full(
                output_shape,
                torch.finfo(src.dtype).max if src.is_floating_point() else torch.iinfo(src.dtype).max,
                device=src.device,
                dtype=src.dtype,
            )
            out.scatter_reduce_(
                0,
                segment_ids.view(-1, *([1] * (src.dim() - 1))).expand_as(src),
                src,
                reduce="amin",
                include_self=False,
            )
            return out

        raise ValueError(f"Unsupported reduce operation: {reduce}")

