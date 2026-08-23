# Security Notes

Benchmarking kit — lowest-risk surface, but still reviewed.

## Controls
- Client targets only operator-configured base URLs (`VLLM_BASE_URL`) — no user-supplied endpoints
- Mock server binds localhost by default and holds no state
- No secrets required; API key read from env only, never logged

## Known gaps
- Benchmark prompts are trusted operator input — do not expose the harness as a multi-user service
