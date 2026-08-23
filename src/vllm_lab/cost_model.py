from __future__ import annotations


def break_even(
    gpu_cost_per_hour: float,
    tokens_per_second_observed: float,
    api_price_per_1m_output: float,
    utilization: float = 0.45,
    gpus: int = 1,
) -> dict:
    """Self-hosted vs API break-even calculator.

    Returns monthly token volume at which self-hosting costs equal the API,
    plus cost curves per million output tokens.
    """
    if tokens_per_second_observed <= 0 or gpu_cost_per_hour <= 0:
        raise ValueError("throughput and gpu cost must be positive")
    seconds_per_month = 730.0 * 3600 * utilization
    monthly_capacity_tokens = tokens_per_second_observed * seconds_per_month * gpus
    monthly_gpu_cost = gpu_cost_per_hour * 730 * gpus
    self_hosted_per_1m = monthly_gpu_cost / (monthly_capacity_tokens / 1_000_000)
    break_even_tokens = monthly_gpu_cost / (api_price_per_1m_output / 1_000_000)
    return {
        "monthly_capacity_tokens": round(monthly_capacity_tokens),
        "self_hosted_usd_per_1m_output": round(self_hosted_per_1m, 4),
        "break_even_tokens_per_month": round(break_even_tokens),
        "break_even_million_tokens": round(break_even_tokens / 1_000_000, 2),
        "assumptions": {
            "utilization": utilization,
            "gpus": gpus,
            "hours_in_month": 730,
        },
    }


def compare_candidates(candidates: list[dict]) -> list[dict]:
    """candidates: [{name, gpu_cost_per_hour, tokens_per_second, gpus}] vs api_price_per_1m_output."""
    rows = []
    for c in candidates:
        row = break_even(
            gpu_cost_per_hour=c["gpu_cost_per_hour"],
            tokens_per_second_observed=c["tokens_per_second"],
            api_price_per_1m_output=c["api_price_per_1m_output"],
            utilization=c.get("utilization", 0.45),
            gpus=c.get("gpus", 1),
        )
        row["name"] = c["name"]
        rows.append(row)
    return sorted(rows, key=lambda r: r["self_hosted_usd_per_1m_output"])
