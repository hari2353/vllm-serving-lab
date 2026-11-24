import threading
import time

import pytest
from fastapi.testclient import TestClient

from vllm_lab.benchmarks.harness import summarize, sweep, to_markdown
from vllm_lab.client import OpenAICompatClient
from vllm_lab.cost_model import break_even, compare_candidates


@pytest.fixture(scope="module")
def mock_base_url():
    from vllm_lab.server_mock import create_mock_app

    app = create_mock_app(tokens_per_second=2000.0)
    return "mock"


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from vllm_lab.server_mock import create_mock_app

    app = create_mock_app(tokens_per_second=5000.0)

    class InlineTransport(httpx.HTTPTransport):
        pass

    c = OpenAICompatClient(base_url="http://mock", model="mock/qwen-0.5b-awq")
    original_post = httpx.post

    def fake_post(url, **kwargs):
        kwargs.pop("headers", None)
        with TestClient(app) as tc:
            resp = tc.post(
                url.replace("http://mock", ""), json=kwargs.get("json"), timeout=30
            )
        import httpx as _hx

        return _hx.Response(
            status_code=resp.status_code,
            json=resp.json(),
            request=_hx.Request("POST", url),
        )

    import vllm_lab.client as client_module

    client_module.httpx.post = fake_post
    yield c
    client_module.httpx.post = original_post


import httpx


def test_chat_parses_usage(client):
    out = client.chat([{"role": "user", "content": "hello world"}], max_tokens=8)
    assert out["completion_tokens"] == 8
    assert out["prompt_tokens"] >= 1
    assert out["content"].count("tok") == 8


def test_summarize_percentiles_and_rates():
    rows = summarize(
        [0.1, 0.2, 0.3, 0.4], [10, 10, 10, 10], wall_seconds=1.0, concurrency=2
    )
    assert rows["requests"] == 4
    assert 0.0 < rows["p50_s"] <= 0.4
    assert rows["tokens_per_second"] == 40.0
    assert rows["requests_per_second"] == 4.0


def test_sweep_against_mock(client):
    prompts = [f"prompt {i} about kubernetes scaling" for i in range(8)]
    rows = sweep(client, prompts, levels=[1, 2], max_tokens=4)
    assert len(rows) == 2
    for row in rows:
        assert row["errors"] == 0
        assert row["requests"] >= 4
    md = to_markdown(rows)
    assert "| conc |" in md


def test_break_even_math():
    result = break_even(
        gpu_cost_per_hour=2.0,
        tokens_per_second_observed=1000.0,
        api_price_per_1m_output=2.0,
        utilization=0.5,
    )
    seconds = 730 * 3600 * 0.5
    capacity = 1000 * seconds
    expected_self_hosted = (2.0 * 730) / (capacity / 1_000_000)
    assert result["self_hosted_usd_per_1m_output"] == pytest.approx(
        expected_self_hosted, rel=0.001
    )
    assert result["break_even_tokens_per_month"] == pytest.approx(
        2.0 * 730 / 2.0 * 1_000_000, rel=0.001
    )


def test_compare_candidates_sorted_by_cost():
    rows = compare_candidates(
        [
            {
                "name": "a100",
                "gpu_cost_per_hour": 3.0,
                "tokens_per_second": 900,
                "api_price_per_1m_output": 2.0,
            },
            {
                "name": "l4",
                "gpu_cost_per_hour": 0.8,
                "tokens_per_second": 300,
                "api_price_per_1m_output": 2.0,
            },
        ]
    )
    assert (
        rows[0]["self_hosted_usd_per_1m_output"]
        <= rows[1]["self_hosted_usd_per_1m_output"]
    )
    assert {r["name"] for r in rows} == {"a100", "l4"}


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        break_even(
            gpu_cost_per_hour=0,
            tokens_per_second_observed=10,
            api_price_per_1m_output=1,
        )
