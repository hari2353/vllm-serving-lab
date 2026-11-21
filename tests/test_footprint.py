import pytest

from vllm_lab.footprint import (
    carbon_per_million_tokens,
    decode_roofline,
    energy_per_million_tokens,
    roofline_efficiency,
)


def test_energy_math_at_ideal_pue():
    out = energy_per_million_tokens(1000, 400, pue=1.0)
    assert out["joules_per_token"] == 0.4
    assert out["kwh_per_1m_tokens"] == 0.111111


def test_pue_and_gpus_scale_energy():
    one = energy_per_million_tokens(1000, 400, pue=1.2)
    two = energy_per_million_tokens(1000, 400, pue=1.2, gpus=2)
    assert two["kwh_per_1m_tokens"] == pytest.approx(one["kwh_per_1m_tokens"] * 2, rel=1e-4)


def test_carbon_math():
    out = carbon_per_million_tokens(1000, 400, pue=1.0, carbon_intensity_g_per_kwh=400)
    assert out["kgco2e_per_1m_tokens"] == 0.044444


def test_energy_rejects_bad_inputs():
    for kwargs in (
        {"tokens_per_second": 0, "gpu_power_w": 400},
        {"tokens_per_second": 100, "gpu_power_w": 0},
        {"tokens_per_second": 100, "gpu_power_w": 400, "gpus": 0},
        {"tokens_per_second": 100, "gpu_power_w": 400, "pue": 0.9},
    ):
        try:
            energy_per_million_tokens(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {kwargs}")


def test_decode_roofline_weights_only():
    out = decode_roofline(7, 2000, bytes_per_param=2.0)
    assert out["bytes_per_token"] == 14_000_000_000
    assert out["roofline_tokens_per_second"] == 142.86


def test_decode_roofline_includes_kv_cache():
    out = decode_roofline(
        7, 2000, bytes_per_param=2.0, kv_bytes_per_token=524_288, context_tokens=4096
    )
    assert out["kv_bytes"] == 2_147_483_648
    assert out["roofline_tokens_per_second"] == round(2000e9 / (14e9 + 2_147_483_648), 2)


def test_quantization_raises_the_ceiling():
    fp16 = decode_roofline(7, 2000, bytes_per_param=2.0)["roofline_tokens_per_second"]
    int4 = decode_roofline(7, 2000, bytes_per_param=0.5)["roofline_tokens_per_second"]
    assert int4 == pytest.approx(fp16 * 4, rel=1e-4)


def test_roofline_efficiency():
    roofline = decode_roofline(7, 2000)
    out = roofline_efficiency(100, roofline)
    assert out["efficiency"] == round(100 / 142.86, 4)
    try:
        roofline_efficiency(100, {"roofline_tokens_per_second": 0})
    except ValueError:
        pass
    else:
        raise AssertionError("zero ceiling accepted")
