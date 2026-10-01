"""Q5 (PLAN.md): DB entry + a 2-log calibration on the customer's unit.

For seeds 0..4: pick one lift_and_drop log and one other log of the STS3215 12 V unit at random.
(i)  start from the STS3215 7.4 V all-logs fit (the DB entry), CMA-ES sigma 0.1, fit on the 2 logs
(ii) start from BAM default values, CMA-ES sigma 0.25 (as every cold fit here), fit on the 2 logs
Evaluate on every other 12 V log; reference = 12 V all-logs fit on the same logs.
"""
import json, os, sys
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from partgap.evaluate import load_logs, make_batch, make_model, per_log_mae
from partgap.fit import fit

F = os.path.join(ROOT, "results", "fits_v2")
PART = "feetech_sts3215_12v"
BUDGET = {"m1": 2000, "m6": 4000}


def run(args):
    m, seed = args
    logs = load_logs(PART)
    rng = np.random.default_rng(seed)
    drops = [i for i, l in enumerate(logs) if l["trajectory"] == "lift_and_drop"]
    others = [i for i, l in enumerate(logs) if l["trajectory"] != "lift_and_drop" and l["mass"] > 0]
    cal = [int(rng.choice(drops)), int(rng.choice(others))]
    rest = [l for i, l in enumerate(logs) if i not in cal]
    cb, rb = make_batch([logs[i] for i in cal]), make_batch(rest)
    db = json.load(open(f"{F}/feetech_sts3215_7v4__{m}__full.json"))["params"]
    p_db, _, _ = fit(PART, m, cb, budget=BUDGET[m], start=db, sigma=0.1, seed=seed)
    p_cold, _, _ = fit(PART, m, cb, budget=BUDGET[m], seed=seed)
    ref = json.load(open(f"{F}/{PART}__{m}__full.json"))["params"]
    r = {"model": m, "seed": seed, "calibration_logs": [logs[i]["filename"] for i in cal],
         "cal_conditions": [{k: logs[i][k] for k in ("mass", "length", "kp", "trajectory")} for i in cal],
         "n_eval": len(rest)}
    for k, p in [("db_plus_2", p_db), ("cold_2", p_cold), ("db_only", db), ("reference", ref)]:
        r[f"{k}_deg"] = float(np.degrees(per_log_mae(make_model(PART, m, p), rb).mean()))
    print(f"{m} seed {seed}: DB+2 {r['db_plus_2_deg']:.2f}  cold+2 {r['cold_2_deg']:.2f}  DB only {r['db_only_deg']:.2f}  "
          f"reference {r['reference_deg']:.2f}", flush=True)
    return r


if __name__ == "__main__":
    with Pool(2) as pool:
        rows = pool.map(run, [(m, s) for m in ["m1", "m6"] for s in range(5)])
    summary = {}
    for m in ["m1", "m6"]:
        rr = [r for r in rows if r["model"] == m]
        med = {k: float(np.median([r[f"{k}_deg"] for r in rr])) for k in ["db_plus_2", "cold_2", "db_only", "reference"]}
        med["ratio_to_reference"] = med["db_plus_2"] / med["reference"]
        med["pass"] = bool(med["db_plus_2"] <= 1.5 * med["reference"] and med["db_plus_2"] < med["cold_2"])
        med["db_beats_cold_in"] = int(sum(r["db_plus_2_deg"] < r["cold_2_deg"] for r in rr))
        summary[m] = med
        print(m, json.dumps(med))
    json.dump({"summary": summary, "runs": rows}, open(os.path.join(ROOT, "results", "q5_calibration.json"), "w"), indent=1)
