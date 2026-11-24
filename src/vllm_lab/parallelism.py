"""Tensor-parallelism memory planning (cross-pollinated from NVIDIA Megatron/TensorRT).

Estimates per-GPU memory for an LLM served with tensor parallelism: weights,
KV cache and activations are sharded across ``tp_size`` GPUs. Pure arithmetic.
"""

from __future__ import annotations


def tp_memory(
    *,
    params: int,
    kv_cache_bytes: int = 0,
    activation_bytes: int = 0,
    tp_size: int = 1,
    bytes_per_param: float = 2.0,
) -> dict:
    """Per-GPU memory for a tensor-parallel deployment."""
    if tp_size < 1:
        raise ValueError("tp_size must be >= 1")
    if params < 0 or kv_cache_bytes < 0 or activation_bytes < 0:
        raise ValueError("byte counts must be >= 0")
    if bytes_per_param <= 0:
        raise ValueError("bytes_per_param must be > 0")

    weight_bytes = params * bytes_per_param
    per_gpu = (weight_bytes + kv_cache_bytes + activation_bytes) / tp_size
    return {
        "tp_size": tp_size,
        "weight_bytes": weight_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "activation_bytes": activation_bytes,
        "per_gpu_bytes": round(per_gpu, 2),
        "total_bytes": weight_bytes + kv_cache_bytes + activation_bytes,
    }
