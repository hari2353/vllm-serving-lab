from __future__ import annotations

import time

from fastapi import FastAPI, Request

MODEL_NAME = "mock/qwen-0.5b-awq"


def create_mock_app(tokens_per_second: float = 500.0, prompt_overhead: int = 9) -> FastAPI:
    app = FastAPI(title="Mock vLLM")
    state = {"requests": 0}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat/completions")
    async def completions(request: Request) -> dict:
        payload = await request.json()
        state["requests"] += 1
        messages = payload.get("messages", [])
        max_tokens = int(payload.get("max_tokens", 16))
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = max(prompt_chars // 4, 1) + prompt_overhead
        time.sleep(max_tokens / tokens_per_second)
        content = " ".join(["tok"] * max_tokens)
        return {
            "model": MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max_tokens,
                "total_tokens": prompt_tokens + max_tokens,
            },
        }

    @app.get("/stats")
    def stats() -> dict:
        return dict(state)

    return app
