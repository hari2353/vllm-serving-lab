from __future__ import annotations

import asyncio
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

MODEL_NAME = "mock/qwen-0.5b-awq"


def create_mock_app(tokens_per_second: float = 500.0, prompt_overhead: int = 9) -> FastAPI:
    app = FastAPI(title="Mock vLLM")
    state = {"requests": 0}

    def usage_for(messages: list[dict], max_tokens: int) -> dict:
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = max(prompt_chars // 4, 1) + prompt_overhead
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": max_tokens,
            "total_tokens": prompt_tokens + max_tokens,
        }

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat/completions")
    async def completions(request: Request):
        payload = await request.json()
        state["requests"] += 1
        messages = payload.get("messages", [])
        max_tokens = int(payload.get("max_tokens", 16))
        usage = usage_for(messages, max_tokens)

        if payload.get("stream"):
            async def events():
                for _ in range(max_tokens):
                    await asyncio.sleep(1 / tokens_per_second)
                    chunk = {"choices": [{"index": 0, "delta": {"content": "tok "}}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        time.sleep(max_tokens / tokens_per_second)
        content = " ".join(["tok"] * max_tokens)
        return {
            "model": MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "length"}],
            "usage": usage,
        }

    @app.get("/stats")
    def stats() -> dict:
        return dict(state)

    return app
