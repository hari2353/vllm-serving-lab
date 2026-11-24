from __future__ import annotations

import math
import statistics
import threading
import time


def _percentile(ordered: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy/Excel default), exact at 0/1."""
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    value = ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return round(value, 4)


def summarize(
    latencies: list[float],
    completion_tokens: list[int],
    wall_seconds: float,
    concurrency: int,
    *,
    ttfts: list[float] | None = None,
    latency_slo_s: float | None = None,
    ttft_slo_s: float | None = None,
    errors: int = 0,
    errors_by_type: dict[str, int] | None = None,
) -> dict:
    ordered = sorted(latencies)
    total_tokens = sum(completion_tokens)
    n = len(latencies)

    def attainment(values: list[float], slo: float | None) -> float | None:
        if slo is None or not values:
            return None
        return sum(1 for v in values if v <= slo) / len(values)

    latency_attainment = attainment(latencies, latency_slo_s)
    ttft_attainment = attainment(ttfts or [], ttft_slo_s)

    if latency_slo_s is None and ttft_slo_s is None:
        goodput = n / wall_seconds if wall_seconds > 0 else 0.0
    else:
        passing = 0
        for i, latency in enumerate(latencies):
            if latency_slo_s is not None and latency > latency_slo_s:
                continue
            if ttft_slo_s is not None:
                if ttfts is None or i >= len(ttfts) or ttfts[i] > ttft_slo_s:
                    continue
            passing += 1
        goodput = passing / wall_seconds if wall_seconds > 0 else 0.0

    ordered_ttfts = sorted(ttfts) if ttfts else []
    total_attempts = n + errors
    return {
        "requests": n,
        "concurrency": concurrency,
        "p50_s": _percentile(ordered, 0.50),
        "p95_s": _percentile(ordered, 0.95),
        "p99_s": _percentile(ordered, 0.99),
        "mean_s": round(statistics.mean(ordered), 4) if ordered else 0.0,
        "ttft_p50_s": _percentile(ordered_ttfts, 0.50) if ordered_ttfts else None,
        "ttft_p95_s": _percentile(ordered_ttfts, 0.95) if ordered_ttfts else None,
        "tokens_per_second": round(total_tokens / wall_seconds, 2)
        if wall_seconds > 0
        else 0.0,
        "requests_per_second": round(n / wall_seconds, 3) if wall_seconds > 0 else 0.0,
        "goodput_rps": round(goodput, 3),
        "latency_slo_attainment": latency_attainment,
        "ttft_slo_attainment": ttft_attainment,
        "errors": errors,
        "error_rate": round(errors / total_attempts, 4) if total_attempts else 0.0,
        "errors_by_type": dict(errors_by_type or {}),
    }


def sweep(
    client,
    prompts: list[str],
    levels: list[int],
    max_tokens: int = 64,
    *,
    stream: bool = False,
    latency_slo_s: float | None = None,
    ttft_slo_s: float | None = None,
    warmup: int = 0,
) -> list[dict]:
    if not prompts:
        msg = "prompts must be non-empty"
        raise ValueError(msg)
    if any(level < 1 for level in levels):
        msg = f"levels must be >= 1, got {levels}"
        raise ValueError(msg)
    if warmup < 0:
        msg = f"warmup must be >= 0, got {warmup}"
        raise ValueError(msg)
    for _ in range(warmup):
        if stream and hasattr(client, "chat_stream"):
            client.chat_stream([{"role": "user", "content": prompts[0]}], max_tokens=1)
        else:
            client.chat([{"role": "user", "content": prompts[0]}], max_tokens=1)

    results = []
    for level in levels:
        latencies: list[float] = []
        tokens: list[int] = []
        ttfts: list[float] = []
        errors_by_type: dict[str, int] = {}
        lock = threading.Lock()

        def worker(prompt: str) -> None:
            try:
                start = time.perf_counter()
                if stream:
                    out = client.chat_stream(
                        [{"role": "user", "content": prompt}], max_tokens=max_tokens
                    )
                else:
                    out = client.chat(
                        [{"role": "user", "content": prompt}], max_tokens=max_tokens
                    )
                latency = time.perf_counter() - start
                with lock:
                    latencies.append(latency)
                    tokens.append(int(out.get("completion_tokens", 0)))
                    if out.get("ttft_s") is not None:
                        ttfts.append(float(out["ttft_s"]))
            except Exception as exc:
                with lock:
                    key = type(exc).__name__
                    errors_by_type[key] = errors_by_type.get(key, 0) + 1

        rounds = max(len(prompts) // level, 1)
        wall_start = time.perf_counter()
        for _ in range(rounds):
            batch = [prompts[i % len(prompts)] for i in range(level)]
            threads = [threading.Thread(target=worker, args=(p,)) for p in batch]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        wall = time.perf_counter() - wall_start
        results.append(
            summarize(
                latencies,
                tokens,
                wall,
                level,
                ttfts=ttfts or None,
                latency_slo_s=latency_slo_s,
                ttft_slo_s=ttft_slo_s,
                errors=sum(errors_by_type.values()),
                errors_by_type=errors_by_type,
            )
        )
    return results


def to_markdown(rows: list[dict]) -> str:
    header = (
        "| conc | req | errors | p50 | p95 | p99 | ttft p95 | tok/s | req/s | goodput |"
        "\n|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in rows:
        ttft = r.get("ttft_p95_s")
        lines.append(
            f"| {r['concurrency']} | {r['requests']} | {r.get('errors', 0)} "
            f"| {r['p50_s']} | {r['p95_s']} | {r['p99_s']} "
            f"| {'-' if ttft is None else ttft} "
            f"| {r['tokens_per_second']} | {r['requests_per_second']} "
            f"| {r.get('goodput_rps', '')} |"
        )
    return "\n".join(lines)
