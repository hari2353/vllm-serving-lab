"""Market-steal additions (2026-11-18): streaming TTFT/ITL, SLO goodput and
error taxonomy, benchmark methodology (warmup + interpolated percentiles).

Compared products treat these as baseline: vLLM benchmark_serving /
LLMPerf / GenAI-Perf report TTFT and ITL, count only SLO-passing requests as
goodput, and exclude warmup requests from tail percentiles.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import vllm_lab.client as client_module
from vllm_lab.benchmarks.harness import summarize, sweep, to_markdown
from vllm_lab.client import OpenAICompatClient
from vllm_lab.server_mock import create_mock_app


# ── streaming client ────────────────────────────────────────────────────────
class _FakeStream:
    def __init__(self, response) -> None:
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def raise_for_status(self) -> None:
        assert self.response.status_code < 400

    def iter_lines(self):
        return iter(self.response.text.splitlines())


@pytest.fixture()
def stream_client(monkeypatch):
    app = create_mock_app(tokens_per_second=5000.0)
    tc = TestClient(app)
    seen = {}

    def fake_stream(method, url, **kwargs):
        payload = dict(kwargs["json"], stream=True)
        seen["payload"] = payload
        return _FakeStream(tc.post(url.replace("http://mock", ""), json=payload))

    monkeypatch.setattr(client_module.httpx, "stream", fake_stream)
    return OpenAICompatClient(base_url="http://mock", model="mock/qwen-0.5b-awq"), seen


class TestStreaming:
    def test_sse_chunks_parse_into_content_and_usage(self, stream_client) -> None:
        client, seen = stream_client
        out = client.chat_stream([{"role": "user", "content": "hello"}], max_tokens=5)
        assert seen["payload"]["stream"] is True
        assert out["content"].count("tok") == 5
        assert out["completion_tokens"] == 5
        assert out["prompt_tokens"] >= 1

    def test_ttft_and_itl_measured(self, stream_client) -> None:
        client, _ = stream_client
        out = client.chat_stream([{"role": "user", "content": "hello"}], max_tokens=5)
        assert out["ttft_s"] >= 0.0
        assert out["mean_itl_s"] >= 0.0
        assert out["n_chunks"] == 5

    def test_mock_stream_ends_with_done(self) -> None:
        tc = TestClient(create_mock_app(tokens_per_second=5000.0))
        body = tc.post(
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
                "stream": True,
            },
        ).text
        assert "data: [DONE]" in body
        assert body.count("data:") >= 3


# ── SLO attainment, goodput, error taxonomy ─────────────────────────────────
class TestSloMetrics:
    def test_attainment_and_goodput(self) -> None:
        row = summarize(
            [0.1, 0.3],
            [10, 10],
            wall_seconds=1.0,
            concurrency=2,
            ttfts=[0.05, 0.2],
            latency_slo_s=0.2,
            ttft_slo_s=0.1,
        )
        assert row["latency_slo_attainment"] == 0.5
        assert row["ttft_slo_attainment"] == 0.5
        assert row["goodput_rps"] == 1.0  # only the request meeting both SLOs
        assert row["ttft_p50_s"] == pytest.approx(0.125)
        assert row["ttft_p95_s"] >= row["ttft_p50_s"]

    def test_no_slo_means_no_fabricated_attainment(self) -> None:
        row = summarize([0.1, 0.2], [10, 10], wall_seconds=1.0, concurrency=2)
        assert row["latency_slo_attainment"] is None
        assert row["ttft_slo_attainment"] is None
        assert row["goodput_rps"] == row["requests_per_second"]

    def test_error_rate_includes_failures_in_denominator(self) -> None:
        row = summarize(
            [0.1],
            [10],
            wall_seconds=1.0,
            concurrency=2,
            errors=1,
            errors_by_type={"TimeoutError": 1},
        )
        assert row["error_rate"] == 0.5
        assert row["errors"] == 1
        assert row["errors_by_type"] == {"TimeoutError": 1}

    def test_zero_samples_is_safe(self) -> None:
        row = summarize(
            [],
            [],
            wall_seconds=0.0,
            concurrency=4,
            errors=2,
            errors_by_type={"ValueError": 2},
        )
        assert row["requests"] == 0
        assert row["error_rate"] == 1.0
        assert row["goodput_rps"] == 0.0


# ── sweep: streaming, warmup, errors, percentiles ───────────────────────────
class TestSweepMethodology:
    def test_streaming_sweep_reports_ttft(self, stream_client) -> None:
        client, _ = stream_client
        rows = sweep(
            client,
            ["a prompt", "another prompt"],
            levels=[1, 2],
            max_tokens=3,
            stream=True,
            latency_slo_s=10.0,
            ttft_slo_s=10.0,
        )
        assert [r["concurrency"] for r in rows] == [1, 2]
        for row in rows:
            assert row["errors"] == 0
            assert row["ttft_p95_s"] is not None
            assert row["goodput_rps"] > 0.0

    def test_warmup_requests_excluded(self) -> None:
        calls = []

        class Stub:
            def chat(self, messages, max_tokens=64):
                calls.append(messages)
                return {"completion_tokens": 1}

        rows = sweep(Stub(), ["p"], levels=[1], max_tokens=1, warmup=2)
        assert len(calls) == 3  # 2 warmup + 1 measured
        assert rows[0]["requests"] == 1

    def test_errors_counted_by_type(self) -> None:
        class Stub:
            def chat(self, messages, max_tokens=64):
                raise TimeoutError("slow")

        rows = sweep(Stub(), ["p"], levels=[1], max_tokens=1)
        row = rows[0]
        assert row["errors"] == 1
        assert row["errors_by_type"] == {"TimeoutError": 1}
        assert row["error_rate"] == 1.0
        assert row["requests"] == 0

    def test_percentiles_use_linear_interpolation(self) -> None:
        row = summarize([0.1, 0.2, 0.3, 0.4], [10] * 4, wall_seconds=1.0, concurrency=1)
        assert row["p50_s"] == pytest.approx(0.25)
        assert row["p99_s"] == pytest.approx(0.397)

    def test_markdown_carries_new_columns(self) -> None:
        row = summarize(
            [0.1],
            [10],
            wall_seconds=1.0,
            concurrency=1,
            ttfts=[0.05],
            latency_slo_s=1.0,
            ttft_slo_s=1.0,
        )
        md = to_markdown([row])
        for column in ("ttft p95", "goodput", "errors"):
            assert column in md

    def test_invalid_sweep_inputs_rejected(self) -> None:
        class Stub:
            def chat(self, messages, max_tokens=64):
                return {"completion_tokens": 1}

        with pytest.raises(ValueError, match="prompts"):
            sweep(Stub(), [], levels=[1])
        with pytest.raises(ValueError, match="levels"):
            sweep(Stub(), ["p"], levels=[0])
        with pytest.raises(ValueError, match="warmup"):
            sweep(Stub(), ["p"], levels=[1], warmup=-1)
