"""Look up a part: simulator settings + how much error to expect + whether you are extrapolating.

    python -m partgap.query list
    python -m partgap.query feetech_sts3215_7v4 --kp 32 --vin 7.4 --inertia 0.012 --gravity-torque 1.6

From Python:
    from partgap.query import lookup
    r = lookup("feetech_sts3215_7v4", kp=32, vin=7.4)
    r["mujoco"]          # gain, forcerange, damping, frictionloss, armature, command_delay, goal_rate_limit
    r["expected_error_deg"], r["warnings"]
"""
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "parts")


def list_parts():
    return sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(DB, "*.json")))


def load_entry(part):
    path = os.path.join(DB, f"{part}.json")
    if not os.path.exists(path):
        raise KeyError(f"unknown part {part!r}; known: {', '.join(list_parts())}")
    return json.load(open(path))


def _nearest(groups, value):
    keys = list(groups)
    return min(keys, key=lambda k: abs(float(k) - value))


def lookup(part, kp, vin=None, inertia=None, gravity_torque=None, model="m1"):
    """Settings for `model` ('m1' native or 'm6' BAM) plus an error estimate and warnings."""
    from .export import mjcf_snippet, mujoco_params_from_entry

    e = load_entry(part)
    c = e["conditions"]
    params = e["identifications"][model]["params"]
    vin = e["part"]["nominal_vin"] if vin is None else vin
    out = {"part": part, "model": model, "kp": kp, "vin": vin, "warnings": []}

    if model == "m1":
        p = mujoco_params_from_entry(e, params, kp=kp, vin=vin)
        out["mujoco"] = p
        out["mjcf"] = mjcf_snippet("joint", p)
        notes = []
        if p["goal_rate_limit"]:
            notes.append(f"firmware limits the goal to {p['goal_rate_limit']:.2f} rad/s: rate-limit ctrl "
                         "(partgap.export.CommandFilter)")
        if p["command_delay"] > 0.001:
            notes.append(f"firmware command delay {1000 * p['command_delay']:.1f} ms: delay ctrl")
        if abs(p["q_offset"]) > 0.005:
            notes.append(f"fitted q_offset {p['q_offset']:.3f} rad belongs to the test bench, not the part: ignore it")
        out["notes"] = notes
    else:
        out["bam_params"] = params
        out["notes"] = ["use with BAM (pip install better-actuator-models[mujoco]): "
                        "bam.model.load_model(<this json>) + bam.mujoco.MujocoController"]

    # expected error: the all-logs fit, at the nearest measured kp
    bykp = e["error"]["by_condition"][model]["kp"]
    k = _nearest(bykp, kp)
    out["expected_error_deg"] = bykp[k]
    out["expected_error_basis"] = f"open-loop pendulum logs at kp={k}, all loads and trajectories"

    tr = e["transfer"]
    kps = c["kp"]
    if kp > max(kps) or kp < min(kps):
        f = tr["higher_kp"][model]["ratio"]
        out["warnings"].append(f"kp {kp} is outside the measured {min(kps)}..{max(kps)}. Holding out the highest "
                               f"kp grew the error x{f:.2f} for this part")
    lo, hi = c["vin"]
    if vin < lo - 0.3 or vin > hi + 0.3:
        out["warnings"].append(f"supply {vin} V is outside the measured {lo}..{hi} V")
    if inertia is not None and inertia > c["load_inertia"][1]:
        f = tr["heavier_load"][model]["ratio"]
        out["warnings"].append(f"load inertia {inertia:g} kg m^2 is above the measured max {c['load_inertia'][1]:.4g}. "
                               f"Holding out the heaviest load grew the error x{f:.2f}")
    if gravity_torque is not None and gravity_torque > c["gravity_torque_max"][1]:
        f = tr["heavier_load"][model]["ratio"]
        out["warnings"].append(f"gravity torque {gravity_torque:g} N m is above the measured max "
                               f"{c['gravity_torque_max'][1]:.3g}. Holding out the heaviest load grew the error x{f:.2f}")
    bd = tr["backdrive"][model]
    out["warnings"].append(
        f"back-driven motion (torque off, or the load driving the joint) is the weak spot: with this entry it is "
        f"{e['error']['by_condition'][model]['phase']['torque_off']:.1f} deg vs {e['error']['by_condition'][model]['phase']['torque_on']:.1f} deg "
        f"powered, and a fit that never saw a drop test was off by {bd['heldout_off_deg']:.0f} deg there. "
        "If your robot falls, is pushed, or runs torque-off, identify your own unit including drop tests "
        "(BAM's recorder): in our test, 2 calibration logs on top of a DB entry still left x1.7-2.2 the error of a full fit")
    if tr.get("other_unit") is None:
        out["warnings"].append("one physical unit measured: unit-to-unit spread is unknown for this part")
    else:
        ou = tr["other_unit"]["all_logs_by_phase_deg"][model]
        out["warnings"].append(
            f"STS3215 unit-to-unit test (7.4 V friction used on a 12 V unit from another lab, motor constants refit): powered "
            f"{ou['friction_only']['torque_on_deg']:.1f} deg vs its own fit {ou['own_insample']['torque_on_deg']:.1f}, "
            f"back-driven {ou['friction_only']['torque_off_deg']:.0f} deg vs {ou['own_insample']['torque_off_deg']:.0f}")
    return out


def main():
    ap = argparse.ArgumentParser(description="PartGap lookup")
    ap.add_argument("part")
    ap.add_argument("--kp", type=float, default=32)
    ap.add_argument("--vin", type=float)
    ap.add_argument("--inertia", type=float, help="load inertia at the output shaft [kg m^2]")
    ap.add_argument("--gravity-torque", type=float, help="max gravity torque on the joint [N m]")
    ap.add_argument("--model", default="m1", choices=["m1", "m6"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.part == "list":
        for p in list_parts():
            e = load_entry(p)
            print(f"{p:22s} {e['part']['manufacturer']} {e['part']['model']} {e['part'].get('variant', '')}")
        return
    r = lookup(a.part, a.kp, a.vin, a.inertia, a.gravity_torque, a.model)
    if a.json:
        print(json.dumps(r, indent=1))
        return
    print(f"{r['part']}  model {r['model']}  kp {r['kp']:g}  vin {r['vin']:g} V")
    if "mjcf" in r:
        print("\nMuJoCo:\n" + r["mjcf"])
    for n in r["notes"]:
        print("note: " + n)
    print(f"\nexpected open-loop position error: {r['expected_error_deg']:.2f} deg ({r['expected_error_basis']})")
    for w in r["warnings"]:
        print("WARNING: " + w)


if __name__ == "__main__":
    main()
