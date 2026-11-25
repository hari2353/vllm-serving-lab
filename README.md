# vLLM Serving Lab

[![ci](https://github.com/hari2353/vllm-serving-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/hari2353/vllm-serving-lab/actions/workflows/ci.yml)

Self-hosted LLM inference engineering kit around **vLLM**: benchmarking harness (latency percentiles +
throughput under concurrency), quantization comparison flow, GPU break-even cost model vs API pricing,
and K8s HPA manifests scaling on vLLM queue-depth metrics.

**Runs anywhere**: a deterministic mock OpenAI-compatible server (`server_mock.py`) powers the whole
benchmark pipeline in CI/laptops with zero GPU; point `VLLM_BASE_URL` at a real vLLM deployment for
production numbers.

## What's inside

| Piece | Purpose |
|---|---|
| `benchmarks/harness.py` | concurrency sweep → p50/p95/p99, TTFT p95, tok/s, req/s, goodput at SLO |
| `client.py` | OpenAI-compatible chat client capturing usage stats |
| `cost_model.py` | self-hosted $/1M output tokens + break-even token volume vs any API price |
| `footprint.py` | operational kWh/kgCO2e per 1M tokens + decode roofline + efficiency |
| `deploy/k8s-hpa.yaml` | Deployment + Service + HPA on `vllm:num_requests_waiting` |
| `server_mock.py` | deterministic mock engine for CI |

## Operational footprint + roofline

Cross-pollinated from Harvard Edge's TinyML-Footprint and the ML systems
roofline model:

```python
from vllm_lab.footprint import carbon_per_million_tokens, decode_roofline, roofline_efficiency

carbon_per_million_tokens(1000, 400, pue=1.15, carbon_intensity_g_per_kwh=400)
# {'joules_per_token': 0.46, 'kwh_per_1m_tokens': 0.127778, 'kgco2e_per_1m_tokens': 0.051111}

roofline = decode_roofline(7, 2000, bytes_per_param=2.0, kv_bytes_per_token=524_288, context_tokens=4096)
roofline_efficiency(measured_tokens_per_second=100, roofline=roofline)
```

Decode at batch 1 is memory-bandwidth bound: weights plus KV cache bytes per
token set the ceiling. Use `bytes_per_param` 2.0 (fp16), 1.0 (int8), or 0.5
(4-bit) to compare quantization. Operational carbon excludes embodied
hardware, matching the TinyML-Footprint boundary.

## Adaptive batching (UC Berkeley RISE Clipper cross-pollination)\n\n`vllm_lab.batching.plan_batches` models Clipper-style adaptive batching:\nwait up to ``max_wait_s`` or until ``max_batch`` requests queue, then process\none batch with a fixed per-batch overhead. Reports p50/p95 latency, mean and\nmax batch size, throughput and a bounded per-request trace:\n\n```python\nfrom vllm_lab.batching import plan_batches\nplan_batches(arrivals, max_batch=8, max_wait_s=0.05, service_overhead_s=0.01)\n```\n\nDeterministic offline planner; no live server needed.\n\n## Tensor parallelism memory (NVIDIA Megatron cross-pollination)

`vllm_lab.parallelism.tp_memory` estimates per-GPU memory for a
tensor-parallel deployment from parameter counts, KV cache and activations.

## KV-cache memory (vllm-project/vllm cross-pollination)

`vllm_lab.kv_cache.kv_cache_bytes` estimates KV-cache memory by layers,
heads, head dim, seq len, batch and TP size.

## Quickstart

```bash
pip install -r requirements.txt

PYTHONPATH=src python -m uvicorn vllm_lab.server_mock:create_mock_app --factory --port 8040 &
export VLLM_BASE_URL=http://127.0.0.1:8040 VLLM_MODEL=mock/qwen-0.5b-awq
pytest -q          # full pipeline incl sweep against the running server
```

Real GPU deployment:

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-0.5B-Instruct-AWQ --quantization awq --max-model-len 8192
export VLLM_BASE_URL=http://127.0.0.1:8000
```

```python
from vllm_lab.cost_model import compare_candidates
print(compare_candidates([
  {"name": "l4",   "gpu_cost_per_hour": 0.8, "tokens_per_second": 300, "api_price_per_1m_output": 2.0},
  {"name": "a100", "gpu_cost_per_hour": 3.0, "tokens_per_second": 900, "api_price_per_1m_output": 2.0},
]))
```

## Serving-grade metrics (market-steal, 2026-11-18)

- **Streaming TTFT/ITL** — `OpenAICompatClient.chat_stream` consumes SSE and
  reports `ttft_s` plus mean inter-token latency; `summarize` adds
  `ttft_p50_s`/`ttft_p95_s`. The mock server streams too, so CI covers the
  parser with zero GPU.
- **SLO goodput** — `sweep(..., latency_slo_s=..., ttft_slo_s=...)` reports
  `latency_slo_attainment`, `ttft_slo_attainment` and `goodput_rps` (only
  requests meeting every supplied SLO, per wall second) plus an
  `error_rate`/`errors_by_type` taxonomy.
- **Methodology** — `warmup=N` requests run before timing and stay out of the
  samples; percentiles use linear interpolation instead of high-biased
  `int(n*p)` indexing.

Production notes from operating this pattern at scale (SageMaker real-time endpoints, ml.g4/g5 instances):
quantize before you scale (AWQ ≈ 3-4× tok/$ vs fp16 at equal p95), size replicas on *waiting-queue*
depth not CPU, and always benchmark with your own prompt-length mix — vendor tok/s numbers assume
short outputs.

MIT
