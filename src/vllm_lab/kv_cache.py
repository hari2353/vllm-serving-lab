"""KV-cache memory planning (cross-pollinated from vllm-project/vllm).

Estimates the memory consumed by the key/value cache for a given model
configuration, batch size and sequence length, across tensor-parallel shards.
Pure arithmetic, no vLLM dependency.
"""

from __future__ import annotations


def kv_cache_bytes(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    bytes_per_element: float = 2.0,
    tp_size: int = 1,
) -> dict:
    if num_layers < 1 or num_kv_heads < 1 or head_dim < 1:
        raise ValueError("num_layers, num_kv_heads and head_dim must be >= 1")
    if seq_len < 1 or batch_size < 1:
        raise ValueError("seq_len and batch_size must be >= 1")
    if bytes_per_element <= 0 or tp_size < 1:
        raise ValueError("bytes_per_element must be > 0 and tp_size >= 1")

    # K and V, per layer: batch * seq * kv_heads * head_dim * bytes
    per_layer = 2 * batch_size * seq_len * num_kv_heads * head_dim * bytes_per_element
    total = per_layer * num_layers
    per_gpu = total / tp_size
    return {
        "num_layers": num_layers,
        "per_layer_bytes": round(per_layer, 2),
        "total_bytes": round(total, 2),
        "per_gpu_bytes": round(per_gpu, 2),
        "tp_size": tp_size,
    }
