"""Hold the servo at 0 rad (encoder 2048, the middle of the range) so you can attach the arm.

    python bench/center.py --servo sts3215 --port /dev/ttyACM0

While it holds, fix the arm on the horn so it hangs straight down (vertical jig) or points along
your reference line (horizontal jig). Press Enter to release. It also prints the position limits:
they should allow +-120 deg around 0 (encoder 683..3413): sin_sin reaches about +-94 deg.
"""
import argparse
import math

from servos import STS3215, XL330


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servo", choices=["sts3215", "xl330"], required=True)
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", type=int, default=1)
    a = ap.parse_args()
    s = (STS3215 if a.servo == "sts3215" else XL330)(a.port, a.id)
    regs = s.registers()
    lo, hi = regs.get("min_position_limit"), regs.get("max_position_limit")
    print(f"position limits (raw): {lo} .. {hi}")
    if isinstance(lo, int) and isinstance(hi, int) and (lo > 683 or (hi and hi < 3413)):
        print("WARNING: limits do not cover +-120 deg. This servo may have been set up for an arm; "
              "reset its limits (or use a new servo) before recording.")
    s.set_goal(0.0)
    s.set_torque(True)
    print(f"holding 0 rad, now at {math.degrees(s.read()['position']):.2f} deg. Attach the arm, then press Enter.")
    input()
    print(f"position {math.degrees(s.read()['position']):.2f} deg")
    s.set_torque(False)


if __name__ == "__main__":
    main()
