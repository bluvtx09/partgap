import pytest

from partgap.query import list_parts, lookup


def test_all_parts_resolve():
    parts = list_parts()
    assert len(parts) == 6
    for p in parts:
        r = lookup(p, kp=16)
        assert r["mujoco"]["gain"] > 0 and r["mujoco"]["forcerange"] > 0
        assert r["expected_error_deg"] > 0
        assert any("back-driven" in w for w in r["warnings"])


def test_extrapolation_warnings():
    r = lookup("dynamixel_xl330", kp=1000, inertia=10.0, gravity_torque=50.0)
    text = " ".join(r["warnings"])
    assert "outside the measured" in text and "load inertia" in text and "gravity torque" in text
    assert "one physical unit" in text


def test_unit_to_unit_note_only_for_sts3215():
    assert any("unit-to-unit test" in w for w in lookup("feetech_sts3215_7v4", kp=32)["warnings"])
    assert not any("unit-to-unit test" in w for w in lookup("dynamixel_mx64", kp=32)["warnings"])


def test_m6_returns_bam_params():
    r = lookup("dynamixel_mx64", kp=32, model="m6")
    assert r["bam_params"]["model"] == "m6" and "mujoco" not in r


def test_unknown_part():
    with pytest.raises(KeyError):
        lookup("nope", kp=1)
