"""Score every fit on its test logs and answer P0, Q0-Q3 of PLAN.md.

python experiments/analyze.py   -> results/analysis.json (+ printed tables)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from partgap.evaluate import load_logs, make_batch, make_model, per_log_mae  # noqa: E402

PARTS = ["dynamixel_mx64", "dynamixel_mx106", "dynamixel_xl330", "feetech_sts3215_7v4",
         "feetech_sts3215_12v", "waveshare_st3025"]
PUBLISHED = {"dynamixel_mx64": "mx64", "dynamixel_mx106": "mx106", "dynamixel_xl330": "xl330",
             "feetech_sts3215_7v4": "feetech_sts3215_7_4V", "waveshare_st3025": "waveshare_st3025"}
FITS = os.path.join(ROOT, "results", "fits")
B0 = {"friction_base": 0.0, "friction_viscous": 0.0}   # start values, no friction
deg = np.degrees


def fitp(part, m, split):
    return json.load(open(os.path.join(FITS, f"{part}__{m}__{split}.json")))


def mae_on(part, model_name, params, logs, names):
    sel = [l for l in logs if l["filename"] in set(names)]
    return per_log_mae(make_model(part, model_name, params), make_batch(sel)), sel


def main():
    out = {"p0": {}, "q0": {}, "q1": {}, "q2": {}, "q3": {}, "by_condition": {}}
    for part in PARTS:
        logs = load_logs(part)
        allnames = [l["filename"] for l in logs]
        for m in ["m1", "m6"]:
            full = fitp(part, m, "full")
            # P0
            mine, _ = mae_on(part, m, full["params"], logs, allnames)
            row = {"mine_deg": float(deg(mine.mean()))}
            if part in PUBLISHED:
                pub = json.load(open(os.path.join(ROOT, "external/bam/bam/params", PUBLISHED[part], f"{m}.json")))
                p, _ = mae_on(part, m, pub, logs, allnames)
                row.update(published_deg=float(deg(p.mean())), ratio=float(mine.mean() / p.mean()),
                           pass_=bool(mine.mean() <= 1.10 * p.mean()))
            out["p0"][f"{part}/{m}"] = row
            # by-condition breakdown of the all-logs fit (for the DB entry)
            bc = {}
            for key, fn in [("kp", lambda l: str(l["kp"])), ("load", lambda l: f"{round(l['mass'], 1):.1f}kg"),
                            ("trajectory", lambda l: l["trajectory"])]:
                groups = {}
                for l, e in zip(logs, mine):
                    groups.setdefault(fn(l), []).append(e)
                bc[key] = {g: round(float(deg(np.mean(v))), 3) for g, v in sorted(groups.items())}
            out["by_condition"][f"{part}/{m}"] = bc
            # Q1
            for split in ["heavy", "kp", "traj"]:
                f = fitp(part, m, split)
                held, _ = mae_on(part, m, f["params"], logs, f["test"])
                ref, _ = mae_on(part, m, full["params"], logs, f["test"])
                out["q1"][f"{part}/{m}/{split}"] = {
                    "n_test": len(f["test"]), "heldout_deg": float(deg(held.mean())), "seen_deg": float(deg(ref.mean())),
                    "ratio": float(held.mean() / ref.mean()), "pass": bool(held.mean() <= 1.5 * ref.mean())}
        # Q0: random split, test logs
        rnd = {m: fitp(part, m, "random") for m in ["m1", "m6"]}
        test = rnd["m1"]["test"]
        b0, _ = mae_on(part, "m1", B0, logs, test)
        q0 = {"n_test": len(test), "b0_deg": float(deg(b0.mean()))}
        for m in ["m1", "m6"]:
            te, _ = mae_on(part, m, rnd[m]["params"], logs, test)
            tr, _ = mae_on(part, m, rnd[m]["params"], logs, rnd[m]["train"])
            q0[f"{m}_deg"] = float(deg(te.mean()))
            q0[f"{m}_train_deg"] = float(deg(tr.mean()))
        out["q0"][part] = q0
        # Q3: temperature vs residual of the all-logs m6 fit
        temps = np.array([l["entries"][0].get("temp", 0.0) for l in logs], dtype=float)
        if temps.max() > 0:
            res, _ = mae_on(part, "m6", fitp(part, "m6", "full")["params"], logs, allnames)
            rho, pv = spearmanr(temps, res)
            out["q3"][part] = {"temp_range": [float(temps.min()), float(temps.max())], "spearman": float(rho),
                               "p": float(pv), "n": len(logs)}
    # Q2: STS3215 7.4V (Rhoban) -> STS3215 12V (other lab, other unit)
    part = "feetech_sts3215_12v"
    logs = load_logs(part)
    test = fitp(part, "m1", "random")["test"]
    b0, _ = mae_on(part, "m1", B0, logs, test)
    for m in ["m1", "m6"]:
        own, _ = mae_on(part, m, fitp(part, m, "random")["params"], logs, test)
        direct, _ = mae_on(part, m, fitp("feetech_sts3215_7v4", m, "full")["params"], logs, test)
        fr, _ = mae_on(part, m, fitp(part, m, "xfer_friction")["params"], logs, test)
        r = {"n_test": len(test), "own_deg": float(deg(own.mean())), "direct_deg": float(deg(direct.mean())),
             "friction_only_deg": float(deg(fr.mean())), "b0_deg": float(deg(b0.mean()))}
        for k in ["direct", "friction_only"]:
            v = r[f"{k}_deg"]
            r[f"{k}_ratio_to_own"] = v / r["own_deg"]
            r[f"{k}_cut_vs_b0"] = 1 - v / r["b0_deg"]
            r[f"{k}_pass"] = bool(v <= 1.5 * r["own_deg"] and (1 - v / r["b0_deg"]) >= 0.5)
        out["q2"][m] = r
    json.dump(out, open(os.path.join(ROOT, "results", "analysis.json"), "w"), indent=1)

    print("P0 (all logs, mine vs published)")
    for k, v in out["p0"].items():
        print(f"  {k:28s} mine {v['mine_deg']:.3f}" + (f"  pub {v['published_deg']:.3f}  x{v['ratio']:.2f}" if "ratio" in v else ""))
    print("Q0 (random split test, deg): B0 / M1 / M6   [train M1 / M6]")
    for k, v in out["q0"].items():
        print(f"  {k:22s} {v['b0_deg']:7.2f} {v['m1_deg']:6.3f} {v['m6_deg']:6.3f}   [{v['m1_train_deg']:.3f} {v['m6_train_deg']:.3f}]")
    print("Q1 (held-out condition / seen, same logs)")
    for k, v in out["q1"].items():
        print(f"  {k:32s} n={v['n_test']:3d} {v['heldout_deg']:.3f} / {v['seen_deg']:.3f} = x{v['ratio']:.2f} {'ok' if v['pass'] else 'FAIL'}")
    print("Q2", json.dumps(out["q2"], indent=1))
    print("Q3", json.dumps(out["q3"], indent=1))


if __name__ == "__main__":
    main()
