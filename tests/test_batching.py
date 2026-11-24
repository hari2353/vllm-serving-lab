from vllm_lab.batching import plan_batches


def test_empty_arrivals():
    out = plan_batches([])
    assert out["requests"] == 0
    assert out["batches"] == 0
    assert out["per_request"] == []


def test_dense_arrivals_fill_max_batch():
    out = plan_batches(
        [0.0, 0.01, 0.02, 0.03],
        max_batch=4,
        max_wait_s=1.0,
        service_overhead_s=0.01,
        per_request_s=0.0,
    )
    assert out["batches"] == 1
    assert out["max_batch_size"] == 4
    assert [row["latency_s"] for row in out["per_request"]] == [0.04, 0.03, 0.02, 0.01]


def test_sparse_arrivals_wait_only_max_wait():
    out = plan_batches(
        [0.0, 1.0],
        max_batch=8,
        max_wait_s=0.05,
        service_overhead_s=0.0,
        per_request_s=0.0,
    )
    assert out["batches"] == 2
    assert out["per_request"][0]["latency_s"] == 0.05
    assert out["per_request"][1]["latency_s"] == 0.0


def test_max_batch_one_serializes_requests():
    out = plan_batches(
        [0.0] * 4,
        max_batch=1,
        max_wait_s=1.0,
        service_overhead_s=0.01,
        per_request_s=0.0,
    )
    assert out["batches"] == 4
    assert out["per_request"][-1]["latency_s"] == 0.04


def test_batching_improves_throughput_under_fixed_overhead():
    arrivals = [0.0] * 8
    batched = plan_batches(
        arrivals,
        max_batch=8,
        max_wait_s=1.0,
        service_overhead_s=0.01,
        per_request_s=0.001,
    )
    serial = plan_batches(
        arrivals,
        max_batch=1,
        max_wait_s=1.0,
        service_overhead_s=0.01,
        per_request_s=0.001,
    )
    assert batched["throughput_rps"] > serial["throughput_rps"]
    assert batched["p95_latency_s"] < serial["p95_latency_s"]


def test_summary_is_bounded():
    out = plan_batches(
        [float(i) for i in range(50)],
        max_batch=1,
        max_wait_s=0.0,
        per_request_s=0.0,
    )
    assert len(out["per_request"]) == 20
    assert out["truncated"] is True


def test_unsorted_arrivals_are_sorted():
    out = plan_batches([0.2, 0.0, 0.1], max_batch=1, max_wait_s=0.0)
    assert [row["arrival_s"] for row in out["per_request"]] == [0.0, 0.1, 0.2]


def test_invalid_inputs_rejected():
    for kwargs in (
        {"max_batch": 0},
        {"max_wait_s": -1},
        {"service_overhead_s": -1},
        {"per_request_s": -1},
    ):
        try:
            plan_batches([0.0], **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {kwargs}")
    try:
        plan_batches([-1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a negative arrival")
