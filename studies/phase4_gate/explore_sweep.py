"""Exploratory (NOT pre-registered; added after run_gate.py failed the gate).
Does the back-driven gap follow 'friction vs gravity' dominance? Sweep the linkage ratio upward (towards a joint that
never falls) and scale both units' friction downward (towards a transparent joint). -> results/sweep.json"""
import json
import numpy as np
from multisim import PART, load_logs, make_batch, rollout, unit_params
from splits import FRICTION_KEYS  # partgap/experiments/splits.py (path set in multisim)

U = unit_params()
logs = [l for l in load_logs(PART) if l["trajectory"] == "lift_and_drop"]   # the only logs with torque-off motion
b = make_batch(logs)
off = np.array([~e["torque_enable"].astype(bool) for e in b["entries"]])


def scaled(p, s):
    return {k: (v * s if k in FRICTION_KEYS and k not in ("dtheta_stribeck", "alpha") else v) for k, v in p.items()}


rows = []
for n, s in [(1, 0.1), (1, 0.25), (1, 0.5), (1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (64, 1)]:
    A, B = scaled(U["A"], s), scaled(U["B"], s)
    gaps, travel = [], []
    for sim_u, real_u in [([A], [B]), ([B], [A])]:
        qs, qr = rollout(b, sim_u, n), rollout(b, real_u, n)
        gaps.append(np.degrees(np.abs(qs - qr))[off].mean())
        travel.append(np.degrees(np.abs(np.diff(qr, axis=0))[off[1:]].sum(axis=0).mean()))
    rows.append({"ratio": n, "friction_scale": s, "backdriven_gap_deg": float(np.mean(gaps)),
                 "fall_travel_deg_per_log": float(np.mean(travel))})
    print(f"n={n:3d} friction x{s:<5}: back-driven gap {np.mean(gaps):7.3f} deg   arm travel while torque off {np.mean(travel):8.1f} deg/log", flush=True)
json.dump(rows, open("results/sweep.json", "w"), indent=1)
