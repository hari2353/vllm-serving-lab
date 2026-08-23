from __future__ import annotations

import json
import os

import httpx


class OpenAICompatClient:
    def __init__(self, base_url: str | None = None, api_key: str = "", model: str = "", timeout: float = 120.0) -> None:
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        self.model = model or os.getenv("VLLM_MODEL", "openai/gpt-oss-20b")
        self.timeout = timeout
        self.last_usage: dict = {}

    def chat(self, messages: list[dict], max_tokens: int = 64, temperature: float = 0.0) -> dict:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        self.last_usage = {
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
        return {
            "content": data["choices"][0]["message"].get("content", ""),
            **self.last_usage,
        }

    def health(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


def save_report(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
