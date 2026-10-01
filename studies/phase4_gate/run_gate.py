"""Phase 4 gate (PLAN.md): unit-to-unit gap per joint architecture. -> results/gate.json"""
import json, sys
import numpy as np
from multisim import PART, load_logs, make_batch, rollout, unit_params

U = unit_params()
A, B = U["A"], U["B"]
ARCH = {  # name: (ratio, kp_scale, [(sim units, real units), (reverse)])
    "S0": (1, 1, [([A], [B]), ([B], [A])]),
    "S1": (1, 4, [([A], [B]), ([B], [A])]),
    "S2": (2, 1, [([A], [B]), ([B], [A])]),
    "S2b": (4, 1, [([A], [B]), ([B], [A])]),
    "S3": (1, 1, [([A, A], [A, B]), ([B, B], [A, B])]),
}


def main(dt_sim):
    logs = load_logs(PART)
    b = make_batch(logs)
    off = np.array([~e["torque_enable"].astype(bool) for e in b["entries"]])
    res = {}
    for name, (n, ks, pairs) in ARCH.items():
        on_e, off_e, all_e, motion = [], [], [], []
        for sim_u, real_u in pairs:
            qs = rollout(b, sim_u, n, ks, dt_sim)
            qr = rollout(b, real_u, n, ks, dt_sim)
            err = np.degrees(np.abs(qs - qr))
            on_e.append(err[~off].mean()); off_e.append(err[off].mean()); all_e.append(err.mean())
            motion.append(np.degrees(np.abs(np.diff(qr, axis=0))[off[1:]].sum(axis=0).mean()))
        res[name] = {"ratio": n, "kp_scale": ks, "powered_deg": float(np.mean(on_e)),
                     "backdriven_deg": float(np.mean(off_e)), "all_deg": float(np.mean(all_e)),
                     "backdriven_travel_deg_per_log": float(np.mean(motion)),
                     "by_direction": {"powered": [float(x) for x in on_e], "backdriven": [float(x) for x in off_e]}}
        print(f"{name:4s} n={n} kp x{ks}: powered {res[name]['powered_deg']:.3f}  back-driven {res[name]['backdriven_deg']:.3f}  "
              f"all {res[name]['all_deg']:.3f}  (real arm travel while torque off: {res[name]['backdriven_travel_deg_per_log']:.1f} deg/log)", flush=True)
    s0 = res["S0"]
    for name, r in res.items():
        r["cut_powered"] = 1 - r["powered_deg"] / s0["powered_deg"]
        r["cut_backdriven"] = 1 - r["backdriven_deg"] / s0["backdriven_deg"]
    res["gate_pass"] = bool(any(res[s]["cut_backdriven"] >= 0.5 for s in ["S2", "S2b", "S3"]))
    return res


if __name__ == "__main__":
    out = {"dt_5ms": main(None), "dt_1ms": main(0.001)}
    json.dump(out, open("results/gate.json", "w"), indent=1)
    for k in out:
        print(k, "gate pass:", out[k]["gate_pass"], {s: (round(out[k][s]["cut_powered"], 2), round(out[k][s]["cut_backdriven"], 2)) for s in ["S1", "S2", "S2b", "S3"]})
