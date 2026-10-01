"""Turn an identified M1 (Coulomb + viscous) entry into native simulator settings.

For a voltage-controlled servo with a firmware P controller (BAM's model):
    duty   = clip(error_gain * error_gain_ratio * kp * (q_target - q), -max_pwm, max_pwm)
    torque = kt / R * vin * duty  -  kt^2 / R * dq
which is exactly a MuJoCo position actuator with
    gain        = error_gain * error_gain_ratio * kp * vin * kt / R      [N m / rad]
    forcerange  = max_pwm * vin * kt / R                                  [N m]
plus joint damping kt^2/R (back-EMF) + viscous friction, and joint frictionloss = Coulomb friction.
(BAM's own to_mujoco puts max_pwm on the gain instead of the force limit; the form above is the exact one.)

Two firmware behaviours have no MuJoCo equivalent and must be applied to ctrl by the caller:
    - command delay (seconds)                         -> `command_delay`
    - goal rate limit (Feetech/Waveshare firmware)    -> `goal_rate_limit` [rad/s], None if absent
`CommandFilter` does both.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "external", "bam"))


def mujoco_params(part, params, kp, vin=None):
    """M1 params (dict) + firmware kp (+ supply voltage) -> MuJoCo settings dict."""
    from .evaluate import make_model
    model = make_model(part, params.get("model", "m1"), params)
    act = model.actuator
    vin = act.vin if vin is None else vin
    kt, R = model.kt.value, model.R.value
    egr = model.error_gain_ratio.value if hasattr(model, "error_gain_ratio") else 1.0
    rate = model.max_velocity.value if hasattr(model, "max_velocity") else None
    return {
        "gain": float(act.error_gain * egr * kp * vin * kt / R),
        "forcerange": float(act.max_pwm * vin * kt / R),
        "damping": float(model.friction_viscous.value + kt ** 2 / R),
        "damping_torque_off": float(model.friction_viscous.value),
        "frictionloss": float(model.friction_base.value),
        "armature": float(model.armature.value),
        "command_delay": float(model.command_delay.value),
        "goal_rate_limit": None if rate is None else float(rate),
        "q_offset": float(model.q_offset.value),
    }


def mjcf_snippet(name, p):
    """XML for a hinge joint + position actuator using the settings above."""
    return (f'<joint name="{name}" type="hinge" damping="{p["damping"]:.6g}" '
            f'frictionloss="{p["frictionloss"]:.6g}" armature="{p["armature"]:.6g}"/>\n'
            f'<position name="{name}_servo" joint="{name}" kp="{p["gain"]:.6g}" '
            f'forcerange="{-p["forcerange"]:.6g} {p["forcerange"]:.6g}" forcelimited="true"/>')


class CommandFilter:
    """Apply firmware command delay and goal rate limit to a stream of goal positions."""

    def __init__(self, p, dt):
        self.delay_steps = p["command_delay"] / dt
        self.rate = p["goal_rate_limit"]
        self.dt = dt
        self.hist = []
        self.target = None

    def __call__(self, goal, q):
        self.hist.append(goal)
        d = self.delay_steps
        k = len(self.hist) - 1
        i0, frac = int(np.floor(d)), d - np.floor(d)
        g0 = self.hist[max(k - i0, 0)]
        g1 = self.hist[max(k - i0 - 1, 0)]
        g = (1 - frac) * g0 + frac * g1
        if self.rate is None:
            return g
        if self.target is None:
            self.target = q
        step = self.rate * self.dt
        self.target = float(np.clip(g, self.target - step, self.target + step))
        return self.target
