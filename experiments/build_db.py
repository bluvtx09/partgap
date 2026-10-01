"""Assemble db/parts/<part>.json from the fits and results/analysis.json, then validate against db/schema.json."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from partgap.evaluate import load_logs  # noqa: E402

G = 9.80665
META = {
    "dynamixel_mx64": dict(manufacturer="ROBOTIS", model="Dynamixel MX-64", variant="", nominal_vin=15.0,
                           bam_actuator="mx64", src="Rhoban BAM raw data", lab="Rhoban (U. Bordeaux)",
                           url="https://huggingface.co/buckets/Gregwar/bam_data/resolve/mx64_raw.tgz"),
    "dynamixel_mx106": dict(manufacturer="ROBOTIS", model="Dynamixel MX-106", variant="", nominal_vin=15.0,
                            bam_actuator="mx106", src="Rhoban BAM raw data", lab="Rhoban (U. Bordeaux)",
                            url="https://huggingface.co/buckets/Gregwar/bam_data/resolve/mx106_raw.tgz"),
    "dynamixel_xl330": dict(manufacturer="ROBOTIS", model="Dynamixel XL330-M288-T", variant="", nominal_vin=7.5,
                            bam_actuator="xl330", src="Rhoban BAM raw data", lab="Rhoban (U. Bordeaux)",
                            url="https://huggingface.co/buckets/Gregwar/bam_data/resolve/xl330_raw.zip"),
    "feetech_sts3215_7v4": dict(manufacturer="Feetech", model="STS3215", variant="7.4 V", nominal_vin=7.4,
                                bam_actuator="sts3215", src="Rhoban BAM raw data", lab="Rhoban (U. Bordeaux)",
                                url="https://huggingface.co/buckets/Gregwar/bam_data/resolve/feetech_sts3215_raw.zip"),
    "feetech_sts3215_12v": dict(manufacturer="Feetech", model="STS3215", variant="12 V (firmware 3.10)", nominal_vin=12.0,
                                bam_actuator="sts3215", src="T-K-233/bam release sts3215-12v-bam-data-v1", lab="T-K-233 (GitHub)",
                                url="https://github.com/T-K-233/bam/releases/tag/sts3215-12v-bam-data-v1"),
    "waveshare_st3025": dict(manufacturer="Waveshare", model="ST3025", variant="12 V", nominal_vin=12.0,
                             bam_actuator="waveshare_st3025", src="duck_mini_pro_headless release st3025-bam-data-v1",
                             lab="i1Cps (GitHub)",
                             url="https://github.com/i1Cps/duck_mini_pro_headless/releases/tag/st3025-bam-data-v1"),
}
SPLIT_NAME = {"heavier_load": "heavy", "higher_kp": "kp", "unseen_trajectory": "traj"}


def r3(x):
    return round(float(x), 3)


def conditions(logs):
    kp = sorted({float(l["kp"]) for l in logs})
    vin = [float(l.get("vin", np.mean([e["input_volts"] for e in l["entries"][:20]]))) for l in logs]
    inertia = [l["mass"] * l["length"] ** 2 + l["arm_mass"] * l["length"] ** 2 / 3 for l in logs]
    grav = [(l["mass"] + l["arm_mass"] / 2) * G * l["length"] for l in logs]
    temps = [l["entries"][0].get("temp", 0.0) for l in logs]
    return {"kp": kp, "vin": [r3(min(vin)), r3(max(vin))],
            "load_inertia": [float(f"{min(inertia):.4g}"), float(f"{max(inertia):.4g}")],
            "gravity_torque_max": [r3(min(grav)), r3(max(grav))],
            "temperature_c": None if max(temps) == 0 else [float(min(temps)), float(max(temps))],
            "trajectories": sorted({l["trajectory"] for l in logs})}


def main():
    a = json.load(open(os.path.join(ROOT, "results", "analysis.json")))
    os.makedirs(os.path.join(ROOT, "db", "parts"), exist_ok=True)
    for part, m in META.items():
        logs = load_logs(part)
        fits = {mm: json.load(open(os.path.join(ROOT, "results", "fits_v2", f"{part}__{mm}__full.json")))
                for mm in ["m1", "m6"]}
        q0 = a["q0"][part]
        other = None
        if part.startswith("feetech_sts3215"):
            q2 = a["q2"]
            other = {"from": "feetech_sts3215_7v4", "to": "feetech_sts3215_12v",
                     "note": "different lab, different unit, 12 V variant; tested on the 12 V held-out logs",
                     **{mm: {"own_fit_deg": r3(q2[mm]["own_deg"]), "direct_deg": r3(q2[mm]["direct_deg"]),
                             "friction_only_deg": r3(q2[mm]["friction_only_deg"]), "b0_deg": r3(q2[mm]["b0_deg"]),
                             "direct_ratio": r3(q2[mm]["direct_ratio_to_own"]),
                             "friction_only_ratio": r3(q2[mm]["friction_only_ratio_to_own"])} for mm in ["m1", "m6"]}}
        entry = {
            "part": {"id": part, "manufacturer": m["manufacturer"], "model": m["model"], "variant": m["variant"],
                     "nominal_vin": m["nominal_vin"], "control": "voltage_p_firmware", "bam_actuator": m["bam_actuator"]},
            "sources": [{"name": m["src"], "url": m["url"], "license": "not stated; see source", "unit": logs[0]["motor"],
                         "lab": m["lab"], "n_logs": len(logs), "bench": "single pendulum (tip mass on an arm), 6 s logs"}],
            "conditions": conditions(logs),
            "identifications": {mm: {"params": fits[mm]["params"], "fitter": f"CMA-ES, {fits[mm]['evals']} evaluations",
                                     "fit_on": "all", "bam_commit": "e9a619d"} for mm in ["m1", "m6"]},
            "error": {"b0_deg": r3(q0["b0_deg"]), "m1_deg": r3(q0["m1_deg"]), "m6_deg": r3(q0["m6_deg"]),
                      "basis": f"random 80/20 split, {q0['n_test']} held-out logs, open-loop position MAE",
                      "by_condition": {mm: a["by_condition"][f"{part}/{mm}"] for mm in ["m1", "m6"]}},
            "transfer": {**{k: {mm: {"ratio": r3(a["q1"][f"{part}/{mm}/{s}"]["ratio"]),
                                     "heldout_deg": r3(a["q1"][f"{part}/{mm}/{s}"]["heldout_deg"])} for mm in ["m1", "m6"]}
                            for k, s in SPLIT_NAME.items()},
                         "other_unit": other},
        }
        json.dump(entry, open(os.path.join(ROOT, "db", "parts", f"{part}.json"), "w"), indent=1)
    try:
        import jsonschema
        schema = json.load(open(os.path.join(ROOT, "db", "schema.json")))
        for part in META:
            jsonschema.validate(json.load(open(os.path.join(ROOT, "db", "parts", f"{part}.json"))), schema)
        print("schema ok")
    except ImportError:
        print("jsonschema not installed; skipped validation")


if __name__ == "__main__":
    main()
