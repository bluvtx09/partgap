"""Run every fit in PLAN.md (resumable). Results: results/fits/<part>__<model>__<split>.json"""
import json, os, sys, time
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np
from partgap.evaluate import load_logs, make_batch
from partgap.fit import fit
from splits import split, FRICTION_KEYS

PARTS = ["dynamixel_mx64", "dynamixel_mx106", "dynamixel_xl330", "feetech_sts3215_7v4",
         "feetech_sts3215_12v", "waveshare_st3025"]
BUDGET = {"m1": 2000, "m6": 4000}
OUT = os.path.join(ROOT, "results", "fits")


def job(args):
    part, m, kind = args
    path = os.path.join(OUT, f"{part}__{m}__{kind}.json")
    if os.path.exists(path):
        return path
    logs = load_logs(part)
    fixed, start = None, None
    if kind == "xfer_friction":   # Q2(c): 7.4V friction fixed, motor constants fitted on 12V train
        src = json.load(open(os.path.join(OUT, f"feetech_sts3215_7v4__{m}__full.json")))["params"]
        fixed = {k: v for k, v in src.items() if k in FRICTION_KEYS}
        tr, te = split(logs, "random")
    else:
        tr, te = split(logs, kind)
    t = time.time()
    params, s, used = fit(part, m, make_batch([logs[i] for i in tr]), budget=BUDGET[m], fixed=fixed, start=start)
    json.dump({"part": part, "model": m, "split": kind, "params": params, "train_mae_rad": s, "evals": used,
               "train": [logs[i]["filename"] for i in tr], "test": [logs[i]["filename"] for i in te],
               "seconds": round(time.time() - t)}, open(path, "w"), indent=1)
    print(f"done {part} {m} {kind} {np.degrees(s):.3f} deg {time.time() - t:.0f}s", flush=True)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(p, m, k) for k in ["full", "random", "heavy", "kp", "traj"] for m in ["m6", "m1"] for p in PARTS]
    with Pool(2) as pool:
        list(pool.imap_unordered(job, jobs))
        list(pool.imap_unordered(job, [("feetech_sts3215_12v", m, "xfer_friction") for m in ["m6", "m1"]]))
    print("ALL DONE", flush=True)
