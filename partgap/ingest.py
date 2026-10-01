"""Raw servo logs -> constant-timestep logs (same resampling as bam.process).

python -m partgap.ingest   writes data/processed/<part>/*.json
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "external", "raw")
OUT = os.path.join(ROOT, "data", "processed")
DT = 0.005

# part id -> (raw folder glob, BAM actuator name, source)
SOURCES = {
    "dynamixel_mx64": ("mx64/**/*.json", "mx64", "Rhoban BAM raw data (HF Gregwar/bam_data)"),
    "dynamixel_mx106": ("mx106/**/*.json", "mx106", "Rhoban BAM raw data (HF Gregwar/bam_data)"),
    "dynamixel_xl330": ("xl330/**/*.json", "xl330", "Rhoban BAM raw data (HF Gregwar/bam_data)"),
    "feetech_sts3215_7v4": ("sts3215/**/*.json", "sts3215", "Rhoban BAM raw data (HF Gregwar/bam_data)"),
    "feetech_sts3215_12v": ("sts3215_12v/**/*.json", "sts3215", "T-K-233/bam release sts3215-12v-bam-data-v1"),
    "waveshare_st3025": ("st3025/**/raw/*.json", "waveshare_st3025", "i1Cps/duck_mini_pro_headless release st3025-bam-data-v1"),
}


def resample(data, dt=DT):
    out = {k: v for k, v in data.items() if k != "entries"}
    if "arm_mass" not in out:
        out["arm_mass"] = out.pop("arm-mass", 0.0)
    out["dt"] = dt
    e = data["entries"]
    ts = np.arange(0.0, e[-1]["timestamp"], dt)
    frame, entries = 0, []
    for t in ts:
        while t > e[frame + 1]["timestamp"]:
            frame += 1
        a, b = e[frame], e[frame + 1]
        w = (t - a["timestamp"]) / (b["timestamp"] - a["timestamp"])
        new = {k: a[k] + (b[k] - a[k]) * w for k in a if k != "timestamp"}
        new["torque_enable"] = bool(new["torque_enable"] > 0.5)
        new["timestamp"] = float(t)
        entries.append(new)
    out["entries"] = entries
    return out


def main():
    for part, (pattern, _, _) in SOURCES.items():
        files = sorted(glob.glob(os.path.join(RAW, pattern), recursive=True))
        os.makedirs(os.path.join(OUT, part), exist_ok=True)
        for f in files:
            d = resample(json.load(open(f)))
            json.dump(d, open(os.path.join(OUT, part, os.path.basename(f)), "w"))
        print(part, len(files))


if __name__ == "__main__":
    main()
