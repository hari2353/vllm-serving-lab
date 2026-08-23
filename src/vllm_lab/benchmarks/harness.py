from __future__ import annotations

import statistics
import threading
import time


def summarize(latencies: list[float], completion_tokens: list[int], wall_seconds: float, concurrency: int) -> dict:
    ordered = sorted(latencies)
    total_tokens = sum(completion_tokens)

    def pct(p: float) -> float:
        if not ordered:
            return 0.0
        idx = min(int(len(ordered) * p), len(ordered) - 1)
        return round(ordered[idx], 4)

    return {
        "requests": len(latencies),
        "concurrency": concurrency,
        "p50_s": pct(0.50),
        "p95_s": pct(0.95),
        "p99_s": pct(0.99),
        "mean_s": round(statistics.mean(ordered), 4) if ordered else 0.0,
        "tokens_per_second": round(total_tokens / wall_seconds, 2) if wall_seconds > 0 else 0.0,
        "requests_per_second": round(len(latencies) / wall_seconds, 3) if wall_seconds > 0 else 0.0,
    }


def sweep(client, prompts: list[str], levels: list[int], max_tokens: int = 64) -> list[dict]:
    results = []
    for level in levels:
        latencies: list[float] = []
        tokens: list[int] = []
        errors = 0
        lock = threading.Lock()

        def worker(prompt: str) -> None:
            nonlocal_errors = None
            try:
                start = time.perf_counter()
                out = client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)
                latency = time.perf_counter() - start
                with lock:
                    latencies.append(latency)
                    tokens.append(out["completion_tokens"])
            except Exception:
                with lock:
                    errors += 1

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
        row = summarize(latencies, tokens, wall, level)
        row["errors"] = sum(1 for _ in range(errors))
        row["errors"] = errors
        results.append(row)
    return results


def to_markdown(rows: list[dict]) -> str:
    header = "| conc | req | p50 | p95 | p99 | tok/s | req/s |\n|---|---|---|---|---|---|---|"
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['concurrency']} | {r['requests']} | {r['p50_s']} | {r['p95_s']} | {r['p99_s']} | {r['tokens_per_second']} | {r['requests_per_second']} |"
        )
    return "\n".join(lines)
