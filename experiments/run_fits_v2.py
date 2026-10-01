"""Fit protocol v2 (PLAN.md, revision 3). Reads results/fits/ (v1), writes results/fits_v2/.

1. full fits: start from the best of {v1 full, every v1 split fit} on ALL logs, polish with CMA-ES (sigma 0.1).
2. split fits: failed if own-train MAE > 1.05 x (v2 full fit on the same train logs).
   Failed ones are refit cold (4 restarts, 2x budget); keep whichever has the lower TRAIN MAE.
3. Q2(c) xfer_friction refit with the v2 STS3215 7.4V friction (cold, 4 restarts, 2x budget), plus the v1
   xfer fit as a candidate; keep the lower train MAE.
"""
import glob
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402

from partgap.evaluate import load_logs, make_batch, make_model, score  # noqa: E402
from partgap.fit import fit  # noqa: E402
from splits import FRICTION_KEYS, split  # noqa: E402

V1 = os.path.join(ROOT, "results", "fits")
V2 = os.path.join(ROOT, "results", "fits_v2")
PARTS = ["dynamixel_mx64", "dynamixel_mx106", "dynamixel_xl330", "feetech_sts3215_7v4",
         "feetech_sts3215_12v", "waveshare_st3025"]
BUDGET = {"m1": 2000, "m6": 4000}


def load(d, part, m, kind):
    return json.load(open(os.path.join(d, f"{part}__{m}__{kind}.json")))


def full_job(args):
    part, m = args
    path = os.path.join(V2, f"{part}__{m}__full.json")
    if os.path.exists(path):
        return
    t = time.time()
    logs = load_logs(part)
    b = make_batch(logs)
    cands = [json.load(open(f))["params"] for f in glob.glob(os.path.join(V1, f"{part}__{m}__*.json"))
             if json.load(open(f))["split"] != "xfer_friction"]
    scored = sorted(((score(make_model(part, m, c), b), i) for i, c in enumerate(cands)))
    start = cands[scored[0][1]]
    params, s, used = fit(part, m, b, budget=BUDGET[m], start=start, sigma=0.1)
    if scored[0][0] < s:           # polishing never makes it worse, but keep the guard
        params, s = start, scored[0][0]
    json.dump({"part": part, "model": m, "split": "full", "params": params, "train_mae_rad": s,
               "evals": used, "start_mae_rad": scored[0][0],
               "train": [l["filename"] for l in logs], "test": [l["filename"] for l in logs]},
              open(path, "w"), indent=1)
    print(f"full {part} {m}: v1 {np.degrees(load(V1, part, m, 'full')['train_mae_rad']):.3f} -> "
          f"start {np.degrees(scored[0][0]):.3f} -> v2 {np.degrees(s):.3f} deg ({time.time() - t:.0f}s)", flush=True)


def split_job(args):
    part, m, kind = args
    path = os.path.join(V2, f"{part}__{m}__{kind}.json")
    if os.path.exists(path):
        return
    v1 = load(V1, part, m, kind)
    logs = load_logs(part)
    tr = [l for l in logs if l["filename"] in set(v1["train"])]
    b = make_batch(tr)
    fixed = None
    if kind == "xfer_friction":
        src = load(V2, "feetech_sts3215_7v4", m, "full")["params"]
        fixed = {k: v for k, v in src.items() if k in FRICTION_KEYS}
        own = score(make_model(part, m, {**v1["params"], **fixed}), b)
        failed = True       # friction source changed: always refit
    else:
        own = score(make_model(part, m, v1["params"]), b)
        ref = score(make_model(part, m, load(V2, part, m, "full")["params"]), b)
        failed = own > 1.05 * ref
    rec = dict(v1)
    rec["v1_train_mae_rad"] = own
    rec["refit"] = False
    if failed:
        t = time.time()
        params, s, used = fit(part, m, b, budget=2 * BUDGET[m], fixed=fixed, restarts=4, seed=1)
        rec["refit"] = True
        rec["refit_train_mae_rad"] = s
        if s < own:
            rec.update(params=params, train_mae_rad=s, evals=used)
        else:
            rec.update(params={**v1["params"], **(fixed or {})}, train_mae_rad=own)
        print(f"refit {part} {m} {kind}: v1 {np.degrees(own):.3f} -> refit {np.degrees(s):.3f} deg "
              f"({time.time() - t:.0f}s)", flush=True)
    json.dump(rec, open(path, "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(V2, exist_ok=True)
    with Pool(2) as pool:
        list(pool.imap_unordered(full_job, [(p, m) for m in ["m6", "m1"] for p in PARTS]))
        jobs = [(p, m, k) for k in ["random", "heavy", "kp", "traj"] for m in ["m6", "m1"] for p in PARTS]
        jobs += [("feetech_sts3215_12v", m, "xfer_friction") for m in ["m6", "m1"]]
        list(pool.imap_unordered(split_job, jobs))
    print("ALL DONE", flush=True)
