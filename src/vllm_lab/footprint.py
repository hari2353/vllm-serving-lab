"""Operational footprint + decode roofline for served LLMs.

Cross-pollinated from harvard-edge/TinyML-Footprint (operational energy and
carbon) and the ML systems roofline model in harvard-edge/cs249r_book.
"""

from __future__ import annotations

JOULES_PER_KWH = 3_600_000.0


def energy_per_million_tokens(
    tokens_per_second: float,
    gpu_power_w: float,
    *,
    gpus: int = 1,
    pue: float = 1.15,
) -> dict:
    """Energy to generate 1M output tokens at the observed throughput.

    ``pue`` is datacenter power-usage effectiveness (1.0 = ideal, 1.15 a
    typical efficient facility). GPU power should be the measured board
    power, not the datasheet TDP."""
    if tokens_per_second <= 0 or gpu_power_w <= 0:
        raise ValueError("throughput and gpu power must be positive")
    if gpus < 1:
        raise ValueError("gpus must be >= 1")
    if pue < 1.0:
        raise ValueError("pue must be >= 1.0")
    joules_per_token = gpus * gpu_power_w * pue / tokens_per_second
    kwh = joules_per_token * 1_000_000 / JOULES_PER_KWH
    return {
        "joules_per_token": round(joules_per_token, 4),
        "kwh_per_1m_tokens": round(kwh, 6),
    }


def carbon_per_million_tokens(
    tokens_per_second: float,
    gpu_power_w: float,
    *,
    carbon_intensity_g_per_kwh: float = 400.0,
    gpus: int = 1,
    pue: float = 1.15,
) -> dict:
    """Operational kgCO2e per 1M output tokens from the energy estimate."""
    if carbon_intensity_g_per_kwh < 0:
        raise ValueError("carbon intensity must be >= 0")
    energy = energy_per_million_tokens(
        tokens_per_second, gpu_power_w, gpus=gpus, pue=pue
    )
    kg = energy["kwh_per_1m_tokens"] * carbon_intensity_g_per_kwh / 1000
    return {
        **energy,
        "carbon_intensity_g_per_kwh": carbon_intensity_g_per_kwh,
        "kgco2e_per_1m_tokens": round(kg, 6),
    }


def decode_roofline(
    params_billions: float,
    memory_bandwidth_gbps: float,
    *,
    bytes_per_param: float = 2.0,
    kv_bytes_per_token: float = 0.0,
    context_tokens: int = 0,
) -> dict:
    """Memory-bandwidth ceiling for single-stream decode (tokens/s).

    Decode reads the weights and the KV cache once per token; compute is
    usually not the limit at batch 1. ``bytes_per_param`` is 2 for fp16,
    ~1 for int8, ~0.5 for 4-bit."""
    if params_billions <= 0 or memory_bandwidth_gbps <= 0:
        raise ValueError("params and bandwidth must be positive")
    if bytes_per_param <= 0:
        raise ValueError("bytes_per_param must be positive")
    if kv_bytes_per_token < 0 or context_tokens < 0:
        raise ValueError("kv bytes and context must be non-negative")
    weight_bytes = params_billions * 1e9 * bytes_per_param
    kv_bytes = kv_bytes_per_token * context_tokens
    bytes_per_token = weight_bytes + kv_bytes
    return {
        "weight_bytes": round(weight_bytes),
        "kv_bytes": round(kv_bytes),
        "bytes_per_token": round(bytes_per_token),
        "roofline_tokens_per_second": round(
            memory_bandwidth_gbps * 1e9 / bytes_per_token, 2
        ),
    }


def roofline_efficiency(measured_tokens_per_second: float, roofline: dict) -> dict:
    """How close the measured decode rate is to the bandwidth ceiling."""
    ceiling = roofline.get("roofline_tokens_per_second", 0.0)
    if ceiling <= 0:
        raise ValueError("roofline must contain a positive ceiling")
    if measured_tokens_per_second < 0:
        raise ValueError("measured throughput must be >= 0")
    return {
        "measured_tokens_per_second": round(measured_tokens_per_second, 2),
        "roofline_tokens_per_second": ceiling,
        "efficiency": round(measured_tokens_per_second / ceiling, 4),
    }
