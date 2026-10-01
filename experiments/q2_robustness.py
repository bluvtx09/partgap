"""Q2 robustness (added after the main analysis; not a pre-registered test).

The pre-registered Q2 test set is only 19 logs. Here every 12 V log is used:
own = 12 V all-logs fit (in-sample, so optimistic -> the ratios below favour transfer less, not more),
direct = 7.4 V all-logs fit, friction_only = 7.4 V friction + motor constants fit on the 12 V train logs.
Also splits the error into torque-on and torque-off (back-driven fall) phases.
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from partgap.evaluate import load_logs, make_batch, make_model, simulate

F = os.path.join(ROOT, "results", "fits_v2")
part = "feetech_sts3215_12v"
logs = load_logs(part)
b = make_batch(logs)
off = np.array([~e["torque_enable"].astype(bool) for e in b["entries"]])
ref = np.array([e["position"] for e in b["entries"]])
out = {}
for m in ["m1", "m6"]:
    cands = {"own_insample": json.load(open(f"{F}/{part}__{m}__full.json"))["params"],
             "direct": json.load(open(f"{F}/feetech_sts3215_7v4__{m}__full.json"))["params"],
             "friction_only": json.load(open(f"{F}/{part}__{m}__xfer_friction.json"))["params"]}
    row = {}
    for k, p in cands.items():
        pos = np.array(simulate.Simulator(make_model(part, m, p)).rollout_log(b, simulate_control=True)[0])
        err = np.degrees(np.abs(pos - ref))
        row[k] = {"all_deg": float(err.mean()), "torque_on_deg": float(err[~off].mean()), "torque_off_deg": float(err[off].mean())}
    out[m] = row
    print(m, " | ".join(f"{k}: all {v['all_deg']:.2f} on {v['torque_on_deg']:.2f} off {v['torque_off_deg']:.2f}" for k, v in row.items()))
json.dump(out, open(os.path.join(ROOT, "results", "q2_robustness.json"), "w"), indent=1)
