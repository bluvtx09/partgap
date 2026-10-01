"""Fit-failure detector (added after seeing the STS3215 12V M1 random-split fit at 1.90 deg on its own
train logs while the all-logs fit scores ~1.2 deg there).

A split fit has failed to converge if, on its OWN train logs, it is worse than the all-logs fit:
    train_mae(split fit) > 1.05 x mae(all-logs fit, same train logs)
(the all-logs fit also saw the test logs, so on the train logs it should be no better than a fit aimed at them).
Prints failures; results/convergence_check.json.
"""
import glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from partgap.evaluate import load_logs, make_batch, make_model, per_log_mae

d = sys.argv[1] if len(sys.argv) > 1 else "results/fits"
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, d, "*.json"))):
    r = json.load(open(f))
    if r["split"] in ("full", "xfer_friction"):
        continue
    full = json.load(open(os.path.join(ROOT, d if os.path.exists(os.path.join(ROOT, d, f"{r['part']}__{r['model']}__full.json")) else "results/fits",
                                       f"{r['part']}__{r['model']}__full.json")))
    logs = [l for l in load_logs(r["part"]) if l["filename"] in set(r["train"])]
    b = make_batch(logs)
    own = per_log_mae(make_model(r["part"], r["model"], r["params"]), b).mean()
    ref = per_log_mae(make_model(r["part"], r["model"], full["params"]), b).mean()
    rows.append({"fit": os.path.basename(f), "own_train_deg": float(np.degrees(own)), "full_on_train_deg": float(np.degrees(ref)),
                 "ratio": float(own / ref), "failed": bool(own > 1.05 * ref)})
    print(f"{rows[-1]['fit']:45s} own {rows[-1]['own_train_deg']:.3f} full {rows[-1]['full_on_train_deg']:.3f} x{rows[-1]['ratio']:.2f}"
          + ("  FAILED" if rows[-1]["failed"] else ""))
json.dump(rows, open(os.path.join(ROOT, "results", "convergence_check.json"), "w"), indent=1)
