from vllm_lab.parallelism import tp_memory


def test_tp_one_equals_full():
    out = tp_memory(params=1000, kv_cache_bytes=200, activation_bytes=50, tp_size=1)
    assert out["per_gpu_bytes"] == 1000 * 2 + 200 + 50


def test_tp_shards_per_gpu_memory():
    base = tp_memory(params=1000, kv_cache_bytes=200, tp_size=1)
    sharded = tp_memory(params=1000, kv_cache_bytes=200, tp_size=4)
    assert sharded["per_gpu_bytes"] == base["per_gpu_bytes"] / 4


def test_total_bytes_invariant_to_tp():
    a = tp_memory(params=1000, tp_size=1)
    b = tp_memory(params=1000, tp_size=8)
    assert a["total_bytes"] == b["total_bytes"]


def test_invalid_inputs():
    cases = [
        {"tp_size": 0},
        {"params": -1},
        {"kv_cache_bytes": -1},
        {"activation_bytes": -1},
        {"bytes_per_param": 0},
    ]
    for kwargs in cases:
        base = {"params": 100}
        base.update(kwargs)
        try:
            tp_memory(**base)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {kwargs}")
