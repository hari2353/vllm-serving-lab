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
| `benchmarks/harness.py` | concurrency sweep → p50/p95/p99, tok/s, req/s tables |
| `client.py` | OpenAI-compatible chat client capturing usage stats |
| `cost_model.py` | self-hosted $/1M output tokens + break-even token volume vs any API price |
| `deploy/k8s-hpa.yaml` | Deployment + Service + HPA on `vllm:num_requests_waiting` |
| `server_mock.py` | deterministic mock engine for CI |

## Quickstart

```bash
pip install -r requirements.txt

python -m uvicorn vllm_lab.server_mock:create_mock_app --factory --port 8040 &
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

Production notes from operating this pattern at scale (SageMaker real-time endpoints, ml.g4/g5 instances):
quantize before you scale (AWQ ≈ 3-4× tok/$ vs fp16 at equal p95), size replicas on *waiting-queue*
depth not CPU, and always benchmark with your own prompt-length mix — vendor tok/s numbers assume
short outputs.

MIT
