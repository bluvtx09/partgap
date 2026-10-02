"""Dry run of the whole study without hardware.

Builds virtual servos with BAM's simulator, records full sessions with the real recording
code (bench/record.py), then runs the real analysis (analysis/run.py, report.py).
The virtual units are made so the answer is known:
  - 5 STS3215 units: friction terms of the PartGap 7.4 V entry scaled by x0.6 .. x1.6,
    dead zone 0..4 ticks  -> unit differences should be detected
  - 3 XL330 units: friction scaled by x0.95 .. x1.05, same dead zone -> should look uniform
Session 2 adds a remount offset (q_offset +-0.5 deg) and new sensor noise.

    python dryrun.py --budget-scale 0.05 --workers 4
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "bench"))
sys.path.insert(0, ROOT)

from servos import SimServo  # noqa: E402
from record import run_session  # noqa: E402

FRICTION = ("friction_base", "friction_stribeck", "load_friction_motor", "load_friction_external",
            "load_friction_motor_stribeck", "load_friction_external_stribeck", "load_friction_motor_quad",
            "load_friction_external_quad", "friction_viscous")
STS_SCALE = {"u1": 0.6, "u2": 0.8, "u3": 1.0, "u4": 1.3, "u5": 1.6}
STS_DZ = {"u1": 0, "u2": 1, "u3": 2, "u4": 3, "u5": 4}
XL_SCALE = {"x1": 0.95, "x2": 1.0, "x3": 1.05}


def unit_params(part, scale, session):
    p = json.load(open(os.path.join(ROOT, "results", "fits_v2", f"{part}__m6__full.json")))["params"]
    p = {k: (v * scale if k in FRICTION else v) for k, v in p.items()}
    if session == 2:
        p["q_offset"] = p.get("q_offset", 0.0) + 0.0087 * (1 if scale >= 1 else -1)
    return p


def record_all(out):
    for unit, sc in STS_SCALE.items():
        for ses in (1, 2):
            sv = SimServo("feetech_sts3215_7v4", "m6", unit_params("feetech_sts3215_7v4", sc, ses),
                          seed=ses * 10 + int(unit[1]), dead_zone=STS_DZ[unit])
            run_session(sv, "sts3215", unit, ses, mass=0.5, arm_mass=0.03, vin=7.4, seller="sim", outdir=out,
                        prompt=False, on_mount=sv.mount)
    for unit, sc in XL_SCALE.items():
        for ses in (1, 2):
            sv = SimServo("dynamixel_xl330", "m6", unit_params("dynamixel_xl330", sc, ses),
                          seed=ses * 10 + int(unit[1]), dead_zone=1)
            run_session(sv, "xl330", unit, ses, mass=0.12, arm_mass=0.012, vin=5.0, seller="sim", outdir=out,
                        prompt=False, on_mount=sv.mount)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-scale", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-record", action="store_true")
    a = ap.parse_args()
    raw = os.path.join(HERE, "dryrun", "raw")
    if not a.skip_record:
        record_all(raw)
    res = os.path.join(HERE, "dryrun", "results.json")
    subprocess.run([sys.executable, os.path.join(HERE, "analysis", "run.py"), "--raw", raw, "--out", res,
                    "--budget-scale", str(a.budget_scale), "--workers", str(a.workers)], check=True)
    subprocess.run([sys.executable, os.path.join(HERE, "analysis", "report.py"), res], check=True)
