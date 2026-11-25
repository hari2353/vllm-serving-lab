from vllm_lab.kv_cache import kv_cache_bytes


def test_basic_kv_cache():
    out = kv_cache_bytes(
        num_layers=2, num_kv_heads=4, head_dim=8, seq_len=16, batch_size=1
    )
    assert out["total_bytes"] == 2 * 2 * 16 * 4 * 8 * 2.0


def test_tp_shards():
    base = kv_cache_bytes(
        num_layers=1, num_kv_heads=4, head_dim=8, seq_len=16, tp_size=1
    )
    sharded = kv_cache_bytes(
        num_layers=1, num_kv_heads=4, head_dim=8, seq_len=16, tp_size=4
    )
    assert sharded["per_gpu_bytes"] == base["per_gpu_bytes"] / 4


def test_invalid_inputs():
    cases = [
        {"num_layers": 0},
        {"num_kv_heads": 0},
        {"head_dim": 0},
        {"seq_len": 0},
        {"batch_size": 0},
        {"bytes_per_element": 0},
        {"tp_size": 0},
    ]
    for kwargs in cases:
        base = {"num_layers": 1, "num_kv_heads": 4, "head_dim": 8, "seq_len": 16}
        base.update(kwargs)
        try:
            kv_cache_bytes(**base)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {kwargs}")
