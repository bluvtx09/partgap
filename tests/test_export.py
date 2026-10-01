"""The MuJoCo position-actuator form must equal BAM's voltage-controlled torque law."""
import json
import os

import numpy as np
import pytest

from partgap.evaluate import make_model
from partgap.export import CommandFilter, mujoco_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = [("feetech_sts3215_7v4", "feetech_sts3215_7_4V", 32, 7.4), ("dynamixel_mx64", "mx64", 16, 15.0),
         ("waveshare_st3025", "waveshare_st3025", 8, 12.0)]


@pytest.mark.parametrize("part,act,kp,vin", CASES)
def test_gain_and_forcerange_match_bam(part, act, kp, vin):
    params = json.load(open(os.path.join(ROOT, "external/bam/bam/params", act, "m1.json")))
    p = mujoco_params(part, params, kp=kp, vin=vin)
    model = make_model(part, "m1", params)
    a = model.actuator
    a.kp, a.vin = kp, vin
    for err, dq in [(1e-3, 0.0), (2e-3, 0.5), (1.0, 0.0), (-1.0, -0.3)]:
        a.reset()
        volts = a.compute_control(err, 0.0, dq, 0.005)
        bam_tau = a.compute_torque(volts, True, 0.0, dq)
        mj_tau = float(np.clip(p["gain"] * err, -p["forcerange"], p["forcerange"]))
        mj_tau -= (p["damping"] - p["damping_torque_off"]) * dq       # back-EMF part of joint damping
        # Feetech/Waveshare firmware rate-limits the target: only compare when the limit is not active
        if p["goal_rate_limit"] and abs(err) > p["goal_rate_limit"] * 0.005:
            continue
        assert mj_tau == pytest.approx(bam_tau, rel=1e-9, abs=1e-12)


def test_command_filter_rate_limit_and_delay():
    p = {"command_delay": 0.010, "goal_rate_limit": 2.0}
    f = CommandFilter(p, 0.005)
    out = [f(1.0, 0.0) for _ in range(6)]
    assert out[0] == pytest.approx(0.01)            # step limited to 2 rad/s * 5 ms
    assert out[5] == pytest.approx(0.06)
    g = CommandFilter({"command_delay": 0.010, "goal_rate_limit": None}, 0.005)
    seq = [g(x, 0.0) for x in [0.0, 1.0, 2.0, 3.0]]
    assert seq == [0.0, 0.0, 0.0, 1.0]              # two-step delay


@pytest.mark.parametrize("part", ["feetech_sts3215_7v4", "dynamixel_mx64", "dynamixel_xl330", "waveshare_st3025"])
def test_entry_export_matches_bam_export(part):
    from partgap.export import mujoco_params_from_entry
    from partgap.query import load_entry
    e = load_entry(part)
    params = e["identifications"]["m1"]["params"]
    a = mujoco_params(part, params, kp=16, vin=e["part"]["nominal_vin"])
    b = mujoco_params_from_entry(e, params, kp=16)
    for k in a:
        assert b[k] == pytest.approx(a[k], rel=1e-12) if a[k] is not None else b[k] is None
