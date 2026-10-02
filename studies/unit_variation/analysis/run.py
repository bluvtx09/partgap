"""All pre-registered analyses of PLAN.md (U1-U5) on the recorded logs.

    python analysis/run.py                       # data/raw -> results/unit_variation.json
    python analysis/run.py --raw dryrun/raw --out dryrun/results.json --budget-scale 0.1 --workers 4

Steps
1. raw logs -> 5 ms logs (partgap.ingest.resample, the same as every PartGap entry)
2. hysteresis of every H log with the field function (business_validation/h1/analyze_h1.py)
3. per unit and session: M1 and M6 fits on the B block (fit protocol v2)
4. U1/U2 transfer ratios, U4 calibration curve, U5 D=32 check
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(STUDY, "..", ".."))
sys.path.insert(0, ROOT)

from partgap.ingest import resample  # noqa: E402
from partgap.evaluate import make_batch, make_model  # noqa: E402
from partgap.fit import fit  # noqa: E402
from bam import simulate  # noqa: E402  (path set by partgap.evaluate)

PART = {"sts3215": "feetech_sts3215_7v4", "xl330": "dynamixel_xl330"}
BUDGET = {"m1": 2000, "m6": 4000}
CAL_ORDER = [  # PLAN.md U4, fixed before measuring
    ("lift_and_drop", "L2", 16), ("sin_time_square", "L2", 16), ("up_and_down", "L1", 32), ("sin_sin", "L1", 8),
    ("lift_and_drop", "L1", 32), ("sin_time_square", "L1", 8), ("up_and_down", "L2", 8), ("sin_sin", "L2", 32),
]
CAL_K = [1, 2, 4, 8]

_spec = importlib.util.spec_from_file_location(
    "analyze_h1", os.path.join(STUDY, "..", "business_validation", "h1", "analyze_h1.py"))
analyze_h1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze_h1)


# ---------------------------------------------------------------- loading
def load(raw_dir):
    """{(motor, unit, session, block): [processed log, ...]} with a 'name' field."""
    out = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, "*", "*", "s*", "*", "*.json"))):
        motor, unit, ses, block = f.split(os.sep)[-5:-1]
        d = json.load(open(f))
        d.pop("registers", None)
        p = resample(d) if block[0] != "H" else d            # H logs keep their raw timing (resampled to 30 Hz below)
        p["name"] = os.path.basename(f)[:-5]
        out.setdefault((motor, unit, int(ses[1:]), block), []).append(p)
    return out


def fit_logs(logs):
    """Strip non-numeric header fields so make_batch only sees what BAM needs."""
    keep = ("mass", "arm_mass", "length", "kp", "vin", "dt", "entries", "name", "trajectory")
    return [{k: l[k] for k in keep if k in l} for l in logs]


# ---------------------------------------------------------------- hysteresis
def hysteresis(log, fps=30):
    e = log["entries"]
    t = np.array([x["timestamp"] for x in e])
    ts = np.arange(t[0], t[-1], 1.0 / fps)
    a = np.degrees(np.interp(ts, t, [x["goal_position"] for x in e]))
    s = np.degrees(np.interp(ts, t, [x["position"] for x in e]))
    items = list(analyze_h1.rests(a, s, fps))
    h = analyze_h1.hyst(items)
    return None if h is None else {"H": h[0], "n_pos": h[1], "n_neg": h[2]}


# ---------------------------------------------------------------- errors
def phase_mae(part, model_name, params, logs):
    """Mean |error| [deg] over all samples, powered samples and torque-off samples."""
    batch = make_batch(fit_logs(logs))
    sim = simulate.Simulator(make_model(part, model_name, params))
    pos = np.array(sim.rollout_log(batch, simulate_control=True)[0])
    ref = np.array([e["position"] for e in batch["entries"]])
    on = np.array([e["torque_enable"] for e in batch["entries"]]).astype(bool)
    err = np.abs(pos - ref)
    err = np.degrees(np.where(np.isfinite(err), err, np.pi))
    per_log = err.mean(axis=0)
    return {"all": float(per_log.mean()),
            "powered": float(err[on].mean()) if on.any() else None,
            "torque_off": float(err[~on].mean()) if (~on).any() else None}


def start_params(motor, m):
    return json.load(open(os.path.join(ROOT, "results", "fits_v2", f"{PART[motor]}__{m}__full.json")))["params"]


def fit_v2(motor, m, logs, scale, seed=0):
    """Fit protocol (PLAN.md, revision 1): from the PartGap entry with sigma 0.1 and with sigma 0.02,
    plus a cold fit (4 restarts, 2x budget); keep the one with the lowest training error."""
    batch = make_batch(fit_logs(logs))
    b = max(40, int(BUDGET[m] * scale))
    start = start_params(motor, m)
    runs = [fit(PART[motor], m, batch, budget=b, start=start, seed=seed, sigma=0.1),
            fit(PART[motor], m, batch, budget=b, start=start, seed=seed + 1, sigma=0.02),
            fit(PART[motor], m, batch, budget=2 * b, restarts=4, seed=seed + 100)]
    p, s, _ = min(runs, key=lambda r: r[1])
    return p, s


def _job_fit(args):
    motor, unit, ses, m, logs, scale = args
    p, s = fit_v2(motor, m, logs, scale)
    print(f"fit {motor} {unit} s{ses} {m}: {np.degrees(s):.2f} deg", flush=True)
    return (motor, unit, ses, m), p


def _job_cal(args):
    i, j, k, cal_logs, start, scale = args
    batch = make_batch(fit_logs(cal_logs))
    b = max(40, int(BUDGET["m1"] * scale))
    runs = [fit(PART["sts3215"], "m1", batch, budget=b, start=start, sigma=0.1),
            fit(PART["sts3215"], "m1", batch, budget=b, start=start, sigma=0.02, seed=1)]
    p, _, _ = min(runs, key=lambda r: r[1])
    return (i, j, k), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(STUDY, "data", "raw"))
    ap.add_argument("--out", default=os.path.join(STUDY, "results", "unit_variation.json"))
    ap.add_argument("--budget-scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    data = load(a.raw)
    res = {"budget_scale": a.budget_scale, "hysteresis": {}, "fits": {}, "transfer": {}, "calibration": {}, "u5": {}}

    # hysteresis
    for (motor, unit, ses, block), logs in data.items():
        if block.startswith("H"):
            res["hysteresis"][f"{motor}/{unit}/{block}"] = hysteresis(logs[0])

    # fits
    units = {}
    for (motor, unit, ses, block) in data:
        units.setdefault(motor, set()).add(unit)
    jobs = [(motor, u, ses, m, data[(motor, u, ses, f"B{ses}")], a.budget_scale)
            for motor in units for u in sorted(units[motor]) for ses in (1, 2) for m in ("m1", "m6")
            if (motor, u, ses, f"B{ses}") in data]
    with Pool(a.workers) as pool:
        fits = dict(pool.map(_job_fit, jobs))
    res["fits"] = {"/".join(map(str, k)): v for k, v in fits.items()}

    # U1 / U2 transfer: T(i->j) = err(theta_i^S1 on j^S2) / err(theta_j^S1 on j^S2)
    for motor in units:
        us = sorted(units[motor])
        for m in ("m1", "m6"):
            for j in us:
                if (motor, j, 2, "B2") not in data:
                    continue
                test = data[(motor, j, 2, "B2")]
                own = phase_mae(PART[motor], m, fits[(motor, j, 1, m)], test)
                for i in us:
                    if i == j:
                        continue
                    other = phase_mae(PART[motor], m, fits[(motor, i, 1, m)], test)
                    res["transfer"][f"{motor}/{m}/{i}->{j}"] = {
                        "own": own, "other": other,
                        **{f"T_{ph}": (other[ph] / own[ph] if own[ph] else None) for ph in ("all", "powered", "torque_off")}}

    # U4 calibration curve (STS, M1)
    if "sts3215" in units:
        us = sorted(units["sts3215"])
        jobs = []
        for j in us:
            by_name = {l["name"]: l for l in data[("sts3215", j, 1, "B1")]}
            cal_all = [by_name[f"{tr}_kp{kp}_{L}"] for tr, L, kp in CAL_ORDER]
            for i in us:
                if i != j:
                    for k in CAL_K:
                        jobs.append((i, j, k, cal_all[:k], fits[("sts3215", i, 1, "m1")], a.budget_scale))
        with Pool(a.workers) as pool:
            cal = dict(pool.map(_job_cal, jobs))
        for j in us:
            test = data[("sts3215", j, 2, "B2")]
            ref = phase_mae(PART["sts3215"], "m1", fits[("sts3215", j, 1, "m1")], test)["all"]
            for i in us:
                if i == j:
                    continue
                for k in CAL_K:
                    e = phase_mae(PART["sts3215"], "m1", cal[(i, j, k)], test)["all"]
                    res["calibration"][f"{i}->{j}/k{k}"] = {"err": e, "ref": ref, "ratio": e / ref}

        # U5: D=32 logs vs the same condition with D=0, both held out from the unit's own session-1 B fit (M6)
        for j in us:
            if ("sts3215", j, 1, "D1") not in data or ("sts3215", j, 2, "B2") not in data:
                continue
            p = fits[("sts3215", j, 1, "m6")]
            d32 = data[("sts3215", j, 1, "D1")]
            d0 = [l for l in data[("sts3215", j, 2, "B2")] if l["name"].endswith("kp16_L2")]
            e32, e0 = phase_mae(PART["sts3215"], "m6", p, d32)["all"], phase_mae(PART["sts3215"], "m6", p, d0)["all"]
            res["u5"][j] = {"err_d32": e32, "err_d0": e0, "ratio": e32 / e0}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
