"""Adaptive request batching with a deadline, following Clipper.

Cross-pollinated from ucbrise/clipper: wait up to ``max_wait_s`` or until
``max_batch`` requests are queued, then process the batch as one unit with
a fixed per-batch overhead. Batching amortizes that overhead across requests,
which raises throughput at the cost of queueing latency. The planner is a
deterministic offline model so a policy can be compared before deploying it.
"""

from __future__ import annotations

from vllm_lab.benchmarks.harness import _percentile

_MAX_ROWS = 20


def plan_batches(
    arrivals: list[float],
    *,
    max_batch: int = 8,
    max_wait_s: float = 0.05,
    service_overhead_s: float = 0.01,
    per_request_s: float = 0.001,
) -> dict:
    """Plan batches for sorted request arrival times and report latencies."""
    if max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    for label, value in (
        ("max_wait_s", max_wait_s),
        ("service_overhead_s", service_overhead_s),
        ("per_request_s", per_request_s),
    ):
        if value < 0:
            raise ValueError(f"{label} must be >= 0")
    times = [float(value) for value in arrivals]
    if any(value < 0 for value in times):
        raise ValueError("arrivals must be >= 0")
    times.sort()

    if not times:
        return {
            "requests": 0,
            "batches": 0,
            "mean_batch_size": 0.0,
            "max_batch_size": 0,
            "p50_latency_s": 0.0,
            "p95_latency_s": 0.0,
            "throughput_rps": 0.0,
            "per_request": [],
            "truncated": False,
        }

    rows: list[dict] = []
    latencies: list[float] = []
    cursor = 0.0
    index = 0
    batch_count = 0
    largest = 0
    while index < len(times):
        deadline = times[index] + max_wait_s
        end = index
        while end < len(times) and end - index < max_batch and times[end] <= deadline:
            end += 1
        size = end - index
        if size == max_batch or end == len(times):
            dispatch = times[end - 1]
        else:
            dispatch = deadline
        start = max(dispatch, cursor)
        completion = start + service_overhead_s + per_request_s * size
        cursor = completion
        batch_count += 1
        largest = max(largest, size)
        for arrival in times[index:end]:
            latency = round(completion - arrival, 4)
            latencies.append(latency)
            if len(rows) < _MAX_ROWS:
                rows.append(
                    {
                        "arrival_s": round(arrival, 4),
                        "batch_size": size,
                        "dispatch_s": round(start, 4),
                        "latency_s": latency,
                    }
                )
        index = end

    ordered = sorted(latencies)
    wall = cursor - times[0]
    return {
        "requests": len(times),
        "batches": batch_count,
        "mean_batch_size": round(len(times) / batch_count, 4),
        "max_batch_size": largest,
        "p50_latency_s": _percentile(ordered, 0.50),
        "p95_latency_s": _percentile(ordered, 0.95),
        "throughput_rps": round(len(times) / wall, 4) if wall > 0 else 0.0,
        "per_request": rows,
        "truncated": len(times) > _MAX_ROWS,
    }
