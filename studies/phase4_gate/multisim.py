"""Pendulum joint driven by k servos through an ideal linkage of ratio n (servo angle = n x output angle).

Same step as BAM's Simulator (firmware control law -> motor torque -> friction budget -> stopping-torque
clipping -> semi-implicit Euler), generalised to several actuators acting on one output through a ratio.
With k = 1 and n = 1 it must reproduce bam.simulate.Simulator exactly (tests in check_equivalence()).
"""
import json
import os
import sys

import numpy as np

PG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PG)
sys.path.insert(0, os.path.join(PG, "experiments"))
from partgap.evaluate import load_logs, make_batch, make_model, simulate  # noqa: E402
from splits import FRICTION_KEYS  # noqa: E402
from bam.testbench import Pendulum  # noqa: E402

FITS = os.path.join(PG, "results", "fits_v2")
PART = "feetech_sts3215_7v4"


def unit_params():
    """A = STS3215 7.4 V (Rhoban) M6. B = same motor constants, friction of the 12 V unit (T-K-233)."""
    a = json.load(open(f"{FITS}/feetech_sts3215_7v4__m6__full.json"))["params"]
    b12 = json.load(open(f"{FITS}/feetech_sts3215_12v__m6__full.json"))["params"]
    a = {**a, "q_offset": 0.0}
    b = {**a, **{k: v for k, v in b12.items() if k in FRICTION_KEYS}}
    return {"A": a, "B": b}


def rollout(batch, unit_list, ratio=1.0, kp_scale=1.0, dt_sim=None):
    """Output angle trajectory (T, N) for k = len(unit_list) servos, each a params dict."""
    dt = float(batch["dt"])
    dt_sim = dt if dt_sim is None else dt_sim
    sub = int(round(dt / dt_sim))
    n = float(ratio)
    k = len(unit_list)
    bench = Pendulum(batch)
    models = []
    for p in unit_list:
        m = make_model(PART, p.get("model", "m6"), p)
        m.actuator.load_log(batch)
        m.actuator.kp = batch["kp"] * kp_scale
        m.reset()
        models.append(m)
    e0 = batch["entries"][0]
    q = np.array(e0["position"], dtype=float)          # start at the logged output angle
    dq = np.array(e0["speed"], dtype=float) if "speed" in e0 else np.zeros_like(q)
    goals = np.stack([e["goal_position"] for e in batch["entries"]])
    delayed = []
    for m in models:
        d = m.command_delay.value
        delayed.append(simulate.fractional_delay_shift(goals, d, dt) if d > 0 else goals)
    out = []
    for t, e in enumerate(batch["entries"]):
        out.append(q.copy())
        enable = e["torque_enable"]
        for _ in range(sub):
            tau_g = bench.compute_bias(q, dq)
            inertia = bench.compute_mass(q, dq)
            net = tau_g.copy()
            budget = np.zeros_like(q)
            for i, m in enumerate(models):
                qs, dqs = n * q, n * dq
                ctrl = m.actuator.compute_control(n * delayed[i][t], qs, dqs, dt_sim)
                tau = m.actuator.compute_torque(ctrl, enable, qs, dqs)
                fl, damp = m.compute_frictions(tau, tau_g / (n * k), dqs)
                net = net + n * tau
                budget = budget + n * (fl + damp * np.abs(dqs))
                inertia = inertia + n * n * m.actuator.get_extra_inertia()
            tau_stop = (inertia / dt_sim) * dq + net
            net = net - np.sign(tau_stop) * np.minimum(np.abs(tau_stop), budget)
            dq = np.clip(dq + net / inertia * dt_sim, -100.0, 100.0)
            q = q + dq * dt_sim
    return np.array(out)


def check_equivalence():
    logs = load_logs(PART)
    b = make_batch(logs)
    p = unit_params()["A"]
    ref = np.array(simulate.Simulator(make_model(PART, "m6", p)).rollout_log(b, simulate_control=True)[0])
    mine = rollout(b, [p])
    return float(np.max(np.abs(ref - mine)))


if __name__ == "__main__":
    print("max |multisim - BAM| (rad):", check_equivalence())
