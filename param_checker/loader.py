from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


@dataclass(frozen=True)
class TensorMeta:
    key: str
    dtype: str
    original_shape: list[int]
    view_shape: list[int]
    numel: int
    min_value: float
    max_value: float


@dataclass(frozen=True)
class TensorView:
    key: str
    dtype: str
    original_shape: list[int]
    view_shape: list[int]
    numel: int
    min_value: float
    max_value: float
    matrix: list[list[float]]


def load_checkpoint(path: str | Path) -> Any:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def flatten_tensors(obj: Any, prefix: str = "") -> Dict[str, torch.Tensor]:
    """Flatten nested structures and collect tensor leaves with dotted keys."""
    output: Dict[str, torch.Tensor] = {}

    if isinstance(obj, torch.Tensor):
        key = prefix or "root"
        output[key] = obj
        return output

    if isinstance(obj, Mapping):
        for raw_key, value in obj.items():
            key_part = str(raw_key)
            full_key = f"{prefix}.{key_part}" if prefix else key_part
            output.update(flatten_tensors(value, full_key))
        return output

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            full_key = f"{prefix}.{idx}" if prefix else str(idx)
            output.update(flatten_tensors(value, full_key))

    return output


def tensor_view_2d(tensor: torch.Tensor) -> torch.Tensor:
    base = tensor.detach().cpu().to(torch.float32)

    if base.ndim == 0:
        return base.reshape(1, 1)
    if base.ndim == 1:
        return base.reshape(1, -1)
    if base.ndim == 2:
        return base
    return base.reshape(base.shape[0], -1)


def tensor_meta(key: str, tensor: torch.Tensor, view: torch.Tensor | None = None) -> TensorMeta:
    view_tensor = view if view is not None else tensor_view_2d(tensor)
    min_value = float(view_tensor.min().item())
    max_value = float(view_tensor.max().item())
    return TensorMeta(
        key=key,
        dtype=str(tensor.dtype),
        original_shape=[int(v) for v in tensor.shape],
        view_shape=[int(v) for v in view_tensor.shape],
        numel=int(tensor.numel()),
        min_value=min_value,
        max_value=max_value,
    )


def tensor_to_view(key: str, tensor: torch.Tensor) -> TensorView:
    view = tensor_view_2d(tensor)
    meta = tensor_meta(key, tensor, view=view)
    return TensorView(
        key=meta.key,
        dtype=meta.dtype,
        original_shape=meta.original_shape,
        view_shape=meta.view_shape,
        numel=meta.numel,
        min_value=meta.min_value,
        max_value=meta.max_value,
        matrix=view.tolist(),
    )
