"""Record one measurement session on one servo (PLAN.md, "측정 순서").

Hardware:
    python bench/record.py --servo sts3215 --port /dev/ttyACM0 --unit u1 --session 1 \
        --mass 0.512 --arm-mass 0.031 --vin 7.4 --seller A

It walks through the blocks of the session and tells you when to move the weight or
turn the jig. Logs go to data/raw/<servo>/<unit>/s<session>/<block>/<name>.json in the
raw BAM format (plus unit, session, config, plane and a register dump).

Dry run without hardware: see dryrun.py (uses servos.SimServo and no prompts).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(STUDY, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "external", "bam"))
sys.path.insert(0, HERE)

from bam.trajectory import trajectories as BAM_TRAJ  # noqa: E402

FIT_TRAJ = ["lift_and_drop", "sin_time_square", "up_and_down", "sin_sin"]
KPS = {"sts3215": [8, 16, 32], "xl330": [100, 200, 300]}
LENGTHS = {"L1": 0.10, "L2": 0.15}
XL_LENGTHS = {"L1": 0.11, "L2": 0.17}                 # close to Rhoban's XL330 bench (0.11/0.14/0.17 m)


class Hysteresis:
    """0 -> +15 deg -> 0 -> -15 deg -> 0 ..., 0.5 s moves, 2 s holds, 6 cycles."""
    amp = np.radians(15.0)
    move, hold, cycles = 0.5, 2.0, 6
    targets = [0.0, 1.0, 0.0, -1.0]
    duration = cycles * len(targets) * (move + hold) + hold

    def __call__(self, t):
        seg = self.move + self.hold
        k = int(t // seg)
        if k >= self.cycles * len(self.targets):
            return 0.0, True
        a = self.targets[(k - 1) % len(self.targets)] * self.amp if k > 0 else 0.0
        b = self.targets[k % len(self.targets)] * self.amp
        u = min(1.0, (t - k * seg) / self.move)
        u = u * u * (3 - 2 * u)                              # smoothstep
        return a + (b - a) * u, True


TRAJ = dict(BAM_TRAJ)
TRAJ["hysteresis"] = Hysteresis()


def record_log(servo, traj_name, kp, config, meta, path):
    """Play one trajectory and save the raw log. Same loop as bam/<maker>/record.py."""
    traj = TRAJ[traj_name]
    servo.setup(config, kp)
    regs = servo.registers()
    goal, torque = traj(0.0)
    servo.set_goal(goal)
    servo.set_torque(torque)
    servo.sleep(1.0)                                         # settle on the first goal, as BAM does

    data = dict(meta, kp=kp, trajectory=traj_name, config=config, registers=regs, entries=[])
    start = servo.now()
    while servo.now() - start < traj.duration:
        t = servo.now() - start
        goal, new_torque = traj(t)
        if new_torque != torque:
            servo.set_torque(new_torque)
            torque = new_torque
        if torque:
            servo.set_goal(goal)
        t0 = servo.now() - start
        e = servo.read()
        t1 = servo.now() - start
        e["timestamp"] = (t0 + t1) / 2.0
        e["goal_position"] = float(goal)
        e["torque_enable"] = bool(torque)
        data["entries"].append(e)

    # slow return to zero, then torque off
    q = data["entries"][-1]["position"]
    servo.set_goal(q)                                        # goal = where the arm is, so re-enabling torque does not snap it
    servo.set_torque(True)
    while abs(q) > 1e-6:
        q = max(0.0, q - 0.01) if q > 0 else min(0.0, q + 0.01)
        servo.set_goal(q)
        servo.sleep(0.01)
    servo.sleep(0.5)
    servo.set_torque(False)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(data, open(path, "w"))
    return data


def session_plan(motor, session):
    """List of (block, config, plane, length_key, kp, trajectory) in recording order."""
    lengths = ["L1", "L2"]
    steps = []
    for L in lengths:
        for kp in KPS[motor]:
            for tr in FIT_TRAJ:
                steps.append((f"B{session}", "B", "vertical", L, kp, tr))
    if motor == "sts3215" and session == 1:
        for tr in FIT_TRAJ:
            steps.append(("D1", "L", "vertical", "L2", 16, tr))
    hcfg = "L" if motor == "sts3215" else "F"
    steps.append((f"H{session}", hcfg, "horizontal", "L1", 16 if motor == "sts3215" else 0, "hysteresis"))
    if motor == "sts3215" and session == 1:
        steps.append(("H1z", "L0", "horizontal", "L1", 16, "hysteresis"))
    return steps


def run_session(servo, motor, unit, session, mass, arm_mass, vin, seller, outdir, prompt=True, on_mount=None):
    lens = LENGTHS if motor == "sts3215" else XL_LENGTHS
    steps = session_plan(motor, session)
    state = None
    udir = os.path.join(outdir, motor, unit)
    os.makedirs(udir, exist_ok=True)
    fpath = os.path.join(udir, "factory_registers.json")
    if not os.path.exists(fpath):                            # first contact with this unit: keep its factory values
        json.dump(servo.registers(), open(fpath, "w"), indent=1)
    servo.factory = json.load(open(fpath))
    first_regs = servo.registers()
    try:
        _run_steps(servo, motor, unit, session, steps, lens, mass, arm_mass, vin, seller, outdir, prompt, on_mount)
    finally:                                                 # also on Ctrl-C or a bus error
        servo.restore_factory()
    json.dump(first_regs, open(os.path.join(outdir, motor, unit, f"s{session}", "registers_at_start.json"), "w"),
              indent=1)


def _run_steps(servo, motor, unit, session, steps, lens, mass, arm_mass, vin, seller, outdir, prompt, on_mount):
    state = None
    for n, (block, config, plane, L, kp, tr) in enumerate(steps):
        if (plane, L) != state:
            msg = (f"\n>>> 지그를 {'수평(서보 축 세로)' if plane == 'horizontal' else '수직'}으로 두고, "
                   f"추를 축에서 {lens[L] * 1000:.0f} mm 구멍에 다세요. 팔이 0 위치에서 "
                   f"{'아래로 늘어지게' if plane == 'vertical' else '어느 방향이든 수평으로'} 두세요. 준비되면 Enter.")
            if prompt:
                input(msg)
            if on_mount:
                on_mount(dict(mass=mass, arm_mass=arm_mass, length=lens[L]), plane == "horizontal")
            state = (plane, L)
        meta = dict(mass=mass, arm_mass=arm_mass, length=lens[L], vin=vin, motor=motor, unit=unit,
                    session=session, block=block, plane=plane, seller=seller)
        path = os.path.join(outdir, motor, unit, f"s{session}", block, f"{tr}_kp{kp}_{L}.json")
        print(f"[{n + 1}/{len(steps)}] {block} {config} {plane} {L} kp={kp} {tr}", flush=True)
        record_log(servo, tr, kp, config, meta, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servo", choices=["sts3215", "xl330"], required=True)
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--unit", required=True, help="u1..u5 or x1..x3")
    ap.add_argument("--session", type=int, choices=[1, 2], required=True)
    ap.add_argument("--mass", type=float, required=True, help="weight [kg]")
    ap.add_argument("--arm-mass", type=float, required=True, help="arm [kg]")
    ap.add_argument("--vin", type=float, required=True)
    ap.add_argument("--seller", required=True)
    ap.add_argument("--outdir", default=os.path.join(STUDY, "data", "raw"))
    a = ap.parse_args()
    from servos import STS3215, XL330
    servo = (STS3215 if a.servo == "sts3215" else XL330)(a.port, a.id)
    run_session(servo, a.servo, a.unit, a.session, a.mass, a.arm_mass, a.vin, a.seller, a.outdir)
    print("끝. 세션", a.session, "완료")


if __name__ == "__main__":
    main()
