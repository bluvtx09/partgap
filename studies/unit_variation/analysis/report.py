"""Apply the pre-registered criteria of PLAN.md to results/unit_variation.json.

    python analysis/report.py [results.json]   -> prints a verdict table and writes <results>.verdicts.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def verdict_T(t, big=1.5, small=1.2):
    if t is None:
        return "데이터 없음"
    return "크다" if t >= big else ("작다" if t <= small else "중간")


def main(path):
    r = json.load(open(path))
    field = json.load(open(os.path.join(HERE, "field_reference.json")))
    out = {}

    # U1, U2
    for motor in ("sts3215", "xl330"):
        for m in ("m6", "m1"):
            rows = [v for k, v in r["transfer"].items() if k.startswith(f"{motor}/{m}/")]
            if rows:
                out[f"T_{motor}_{m}"] = {ph: med([x[f"T_{ph}"] for x in rows]) for ph in ("all", "powered", "torque_off")}
                out[f"T_{motor}_{m}"]["n_pairs"] = len(rows)
    t_sts = out.get("T_sts3215_m6", {}).get("all")
    t_xl = out.get("T_xl330_m6", {}).get("all")
    out["U1"] = verdict_T(t_sts)
    out["U2"] = ("데이터 없음" if t_xl is None else
                 "Dynamixel이 더 균일" if (t_xl <= 1.2 and (t_sts is None or t_xl < t_sts)) else "아님")

    # U3
    h = r["hysteresis"]
    h1 = {k.split("/")[1]: v["H"] for k, v in h.items() if k.startswith("sts3215/") and k.endswith("/H1") and v}
    h2 = {k.split("/")[1]: v["H"] for k, v in h.items() if k.startswith("sts3215/") and k.endswith("/H2") and v}
    h1z = {k.split("/")[1]: v["H"] for k, v in h.items() if k.startswith("sts3215/") and k.endswith("/H1z") and v}
    if len(h1) >= 2:
        rng = float(np.ptp(list(h1.values())))
        retest = med([abs(h1[u] - h2[u]) for u in h1 if u in h2])
        ref = field[str(len(h1))] if str(len(h1)) in field else field["5"]
        if retest is not None and rng <= 2 * retest:
            v = "측정 잡음과 구분 안 됨"
        elif rng < ref["p25"]:
            v = "같은 설정에서는 현장보다 차이가 작다 (설정·조립 탓이 큼)"
        elif rng >= ref["p50"]:
            v = "제조 편차만으로 현장 차이가 재현된다"
        else:
            v = "둘 다 기여한다"
        out["U3"] = {"H1_by_unit_deg": h1, "range_deg": rng, "retest_median_deg": retest,
                     "field_p25": ref["p25"], "field_p50": ref["p50"], "verdict": v,
                     "dead_zone_effect_deg": med([h1[u] - h1z[u] for u in h1 if u in h1z])}
    else:
        out["U3"] = "데이터 없음"

    # U4
    cal = r["calibration"]
    curve = {}
    for k in (1, 2, 4, 8):
        rat = [v["ratio"] for key, v in cal.items() if key.endswith(f"/k{k}")]
        if rat:
            curve[k] = med(rat)
    need = next((k for k in sorted(curve) if curve[k] <= 1.2), None)
    out["U4"] = {"median_ratio_by_k": curve,
                 "verdict": ("데이터 없음" if not curve else
                             f"로그 {need}개면 충분" if need else "짧은 키트로는 부족 (k=8에서도 1.2 초과)")}

    # U5
    u5 = r["u5"]
    if u5:
        rr = med([v["ratio"] for v in u5.values()])
        out["U5"] = {"median_ratio": rr, "verdict": "D=32에서 다시 재야 함" if rr > 1.5 else "D=0 값으로 충분"}

    json.dump(out, open(path.replace(".json", ".verdicts.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(STUDY, "results", "unit_variation.json"))
