"""Q4: does the MuJoCo export of an M1 entry reproduce BAM's own M1 simulation?

For each part: take the M1 parameters fit on all logs, build a MuJoCo pendulum per log
(same mass, arm mass, length as the test bench), drive it with partgap.export settings,
and compare the position MAE against the BAM simulator with the same parameters.
Pass (PLAN.md): MuJoCo MAE <= 1.2 x BAM MAE. Also re-run MuJoCo at 1 ms to see numerical error.

python experiments/q4_mujoco_export.py [params_source]   params_source: fits (default) | published
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from partgap.evaluate import load_logs, make_batch, make_model, per_log_mae  # noqa: E402
from partgap.export import CommandFilter, mujoco_params  # noqa: E402

PARTS = ["dynamixel_mx64", "dynamixel_mx106", "dynamixel_xl330", "feetech_sts3215_7v4",
         "feetech_sts3215_12v", "waveshare_st3025"]
PUBLISHED = {"dynamixel_mx64": "mx64", "dynamixel_mx106": "mx106", "dynamixel_xl330": "xl330",
             "feetech_sts3215_7v4": "feetech_sts3215_7_4V", "waveshare_st3025": "waveshare_st3025"}


def pendulum_xml(log, p, dt):
    m, am, L = log["mass"], log["arm_mass"], log["length"]
    M = m + am
    off = p["q_offset"]
    if M <= 0 or L <= 0:
        M, c, Icom = 1e-6, 0.0, 1e-9
    else:
        c = (m * L + am * L / 2) / M
        Ipivot = m * L ** 2 + am * L ** 2 / 3
        Icom = max(Ipivot - M * c ** 2, 1e-9)
    pos = f"{-c * np.sin(off):.9g} 0 {-c * np.cos(off):.9g}"
    return f"""
<mujoco>
  <option timestep="{dt}" gravity="0 0 -9.80665"/>
  <worldbody>
    <body name="arm">
      <joint name="j" type="hinge" axis="0 1 0" damping="{p['damping']}" frictionloss="{p['frictionloss']}" armature="{p['armature']}"/>
      <inertial pos="{pos}" mass="{M}" diaginertia="{Icom} {Icom} {Icom}"/>
    </body>
  </worldbody>
  <actuator>
    <position name="servo" joint="j" kp="{p['gain']}" forcerange="{-p['forcerange']} {p['forcerange']}" forcelimited="true"/>
  </actuator>
</mujoco>"""


def rollout_mujoco(log, params, dt_sim):
    p = mujoco_params(log["part"], params, kp=log["kp"], vin=log.get("vin"))
    dt_log = log["dt"]
    model = mujoco.MjModel.from_xml_string(pendulum_xml(log, p, dt_sim))
    data = mujoco.MjData(model)
    sub = int(round(dt_log / dt_sim))
    e0 = log["entries"][0]
    data.qpos[0], data.qvel[0] = e0["position"], e0.get("speed", 0.0)
    filt = CommandFilter(p, dt_log)
    out = []
    for e in log["entries"]:
        out.append(data.qpos[0])
        on = bool(e["torque_enable"])
        model.actuator_gainprm[0, 0] = p["gain"] if on else 0.0
        model.actuator_biasprm[0, 1] = -p["gain"] if on else 0.0
        model.dof_damping[0] = p["damping"] if on else p["damping_torque_off"]
        data.ctrl[0] = filt(e["goal_position"], float(data.qpos[0]))
        for _ in range(sub):
            mujoco.mj_step(model, data)
    ref = np.array([e["position"] for e in log["entries"]])
    return float(np.mean(np.abs(np.array(out) - ref)))


def main(source="fits"):
    rows = {}
    for part in PARTS:
        if source == "published":
            if part not in PUBLISHED:
                continue
            params = json.load(open(os.path.join(ROOT, "external/bam/bam/params", PUBLISHED[part], "m1.json")))
        else:
            params = json.load(open(os.path.join(ROOT, "results/fits_v2", f"{part}__m1__full.json")))["params"]
        logs = load_logs(part)
        for l in logs:
            l["part"] = part
        bam = per_log_mae(make_model(part, "m1", params), make_batch(logs))
        mj5 = np.array([rollout_mujoco(l, params, 0.005) for l in logs])
        mj1 = np.array([rollout_mujoco(l, params, 0.001) for l in logs])
        r = {"n": len(logs), "bam_deg": float(np.degrees(bam.mean())), "mujoco_5ms_deg": float(np.degrees(mj5.mean())),
             "mujoco_1ms_deg": float(np.degrees(mj1.mean())),
             "ratio_5ms": float(mj5.mean() / bam.mean()), "ratio_1ms": float(mj1.mean() / bam.mean()),
             "per_log_ratio_p90_5ms": float(np.percentile(mj5[bam > 1e-6] / bam[bam > 1e-6], 90)),
             "pass": bool(mj5.mean() <= 1.2 * bam.mean())}
        rows[part] = r
        print(f"{part:22s} BAM {r['bam_deg']:.3f}  MuJoCo 5ms {r['mujoco_5ms_deg']:.3f} (x{r['ratio_5ms']:.2f})  "
              f"1ms {r['mujoco_1ms_deg']:.3f} (x{r['ratio_1ms']:.2f})  pass={r['pass']}", flush=True)
    if source == "fits":
        json.dump(rows, open(os.path.join(ROOT, "results", "q4_mujoco_export.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
