"""Servo drivers for the unit-variation bench.

All drivers share one small interface used by record.py:

    setup(config, kp)          write the firmware settings for this log
    set_goal(rad), set_torque(bool)
    read() -> dict             position [rad], speed [rad/s], load, input_volts, temp, current
    registers() -> dict        every register, for the log header
    now(), sleep(s)            clock (wall clock for hardware, virtual for the simulator)

Hardware access goes through rustypot (the library Rhoban used to record the BAM logs),
so positions and gains are in the same units as the PartGap entries.
`SimServo` replaces the hardware with BAM's simulator for a dry run of the whole pipeline.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# Settings per configuration name (see PLAN.md, "펌웨어 설정")
STS_CONFIGS = {
    "B": dict(d=0, i=0),                                   # Rhoban / BAM fitting setting, P = kp
    "L": dict(p=16, d=32, i=0, acceleration=254, return_delay=0),   # LeRobot SO-100/101 setting
    "L0": dict(p=16, d=32, i=0, acceleration=254, return_delay=0, dead_zone=0),
}
XL_CONFIGS = {
    "B": dict(d=0, i=0),
    "F": dict(),                                            # factory PID
}


def _one(x):
    """rustypot returns a list for some reads and a scalar for others."""
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


class _Hardware:
    def __init__(self, port, servo_id=1, baudrate=1_000_000):
        self.id = servo_id
        self.port = port
        self.c = self._controller(port, baudrate)
        self.factory = None          # set by record.run_session from factory_registers.json

    def now(self):
        return time.time()

    def sleep(self, s):
        time.sleep(s)

    def registers(self):
        out = {}
        for r in self.c.registers():
            name = r.name if hasattr(r, "name") else r["name"]
            try:
                out[name] = int(self.c.read_register(self.id, name, retries=2))
            except Exception as e:        # some write-only or factory registers refuse reads
                out[name] = f"unreadable: {e!r}"[:80]
        return out


class STS3215(_Hardware):
    motor = "sts3215"

    def _controller(self, port, baudrate):
        import rustypot
        return rustypot.Sts3215PyController(port, baudrate, 0.1)

    def setup(self, config, kp):
        s = STS_CONFIGS[config]
        c, i = self.c, self.id
        c.write_torque_enable(i, False)
        c.write_lock(i, False)                                   # unlock EEPROM writes
        c.write_p_coefficient(i, int(s.get("p", kp)))
        c.write_d_coefficient(i, int(s["d"]))
        c.write_i_coefficient(i, int(s["i"]))
        if "acceleration" in s:
            c.write_maximum_acceleration(i, s["acceleration"])
            c.write_acceleration(i, s["acceleration"])
        if "return_delay" in s:
            c.write_return_delay_time(i, s["return_delay"])
        # dead zone: 0 for L0, otherwise the unit's own factory value (EEPROM keeps whatever was written last)
        cw = s.get("dead_zone", self.factory["cw_dead_zone"])
        ccw = s.get("dead_zone", self.factory["ccw_dead_zone"])
        c.write_cw_dead_zone(i, int(cw))
        c.write_ccw_dead_zone(i, int(ccw))
        c.write_lock(i, True)

    def restore_factory(self):
        """P, D, I and dead zone back to the values read on first contact."""
        c, i, f = self.c, self.id, self.factory
        c.write_torque_enable(i, False)
        c.write_lock(i, False)
        c.write_p_coefficient(i, int(f["p_coefficient"]))
        c.write_d_coefficient(i, int(f["d_coefficient"]))
        c.write_i_coefficient(i, int(f["i_coefficient"]))
        c.write_cw_dead_zone(i, int(f["cw_dead_zone"]))
        c.write_ccw_dead_zone(i, int(f["ccw_dead_zone"]))
        c.write_lock(i, True)

    def set_goal(self, q):
        self.c.write_goal_position(self.id, float(q))

    def set_torque(self, on):
        self.c.write_torque_enable(self.id, bool(on))

    def read(self):
        c, i = self.c, self.id
        return {
            "position": float(_one(c.read_present_position(i))),
            "speed": float(_one(c.read_present_speed(i))),
            "load": float(_one(c.read_present_load(i))),
            "input_volts": float(_one(c.read_present_voltage(i))) * 0.1,
            "temp": float(_one(c.read_present_temperature(i))),
        }


class XL330(_Hardware):
    motor = "xl330"
    RAD_S_PER_COUNT = 0.229 * 2 * np.pi / 60
    PWM_LIMIT = 885

    def _controller(self, port, baudrate):
        import rustypot
        return rustypot.Xl330PyController(port, baudrate, 0.1)

    def setup(self, config, kp):
        s = XL_CONFIGS[config]
        c, i = self.c, self.id
        c.write_torque_enable(i, False)
        c.write_operating_mode(i, 3)                         # position control
        f = self.factory
        if config == "B":
            p, d, ii = kp, s["d"], s["i"]
        else:                                                # F: power-on PID read on first contact
            p, d, ii = f["position_p_gain"], f["position_d_gain"], f["position_i_gain"]
        c.write_position_p_gain(i, int(p))
        c.write_position_d_gain(i, int(d))
        c.write_position_i_gain(i, int(ii))
        c.write_profile_velocity(i, 0)                       # no firmware profile, as in the BAM logs
        c.write_profile_acceleration(i, 0)

    def restore_factory(self):
        self.setup("F", None)

    def set_goal(self, q):
        self.c.write_goal_position(self.id, float(q))

    def set_torque(self, on):
        self.c.write_torque_enable(self.id, bool(on))

    def read(self):
        c, i = self.c, self.id
        pwm = float(_one(c.read_present_pwm(i)))
        if pwm > 2**15 - 1:
            pwm -= 2**16
        return {
            "position": float(_one(c.read_present_position(i))),
            "speed": float(_one(c.read_present_velocity(i))) * self.RAD_S_PER_COUNT,
            "load": float(np.clip(pwm / self.PWM_LIMIT, -1, 1)),
            "input_volts": float(_one(c.read_present_input_voltage(i))) / 10.0,
            "temp": float(_one(c.read_present_temperature(i))),
        }


class SimServo:
    """A virtual servo driven by BAM's simulator, with a virtual clock.

    `params` are BAM model parameters (one virtual unit). The bus round trip is modelled
    as `bus_dt` of virtual time per read, so the log timing looks like a real recording.
    `dead_zone` (encoder ticks) is applied to the position error like the firmware does,
    so the hysteresis pipeline can be checked; BAM itself has no dead zone.
    """

    TICK = 2 * np.pi / 4096

    def __init__(self, part, model_name, params, bus_dt=0.003, sub_dt=0.001, noise_ticks=0.5, seed=0,
                 dead_zone=1):
        sys.path.insert(0, ROOT)
        from partgap.evaluate import make_model
        from partgap.ingest import SOURCES
        self.motor = SOURCES[part][1]
        self.model = make_model(part, model_name, params)
        self.part = part
        self.bus_dt, self.sub_dt = bus_dt, sub_dt
        self.rng = np.random.default_rng(seed)
        self.noise_ticks = noise_ticks
        self.factory_dead_zone = dead_zone
        self.dead_zone = dead_zone
        self.t = 0.0
        self.mount(dict(mass=0.0, arm_mass=0.0, length=0.0), horizontal=False)

    # bench side ----------------------------------------------------------
    def mount(self, load, horizontal):
        from bam import simulate
        log = {"mass": load["mass"], "arm_mass": load["arm_mass"], "length": load["length"],
               "kp": 32, "vin": self.model.actuator.vin, "dt": self.sub_dt, "entries": []}
        self.model.actuator.load_log(log)
        if horizontal:
            self.model.actuator.testbench.compute_bias = lambda q, dq: 0.0
        self.sim = simulate.Simulator(self.model)
        self.sim.reset(0.0, 0.0)
        self.goal, self.torque = 0.0, False

    # driver interface ------------------------------------------------------
    def now(self):
        return self.t

    def sleep(self, s):
        self._advance(s)

    def registers(self):
        return {"simulated": True, "cw_dead_zone": self.dead_zone, "ccw_dead_zone": self.dead_zone}

    def restore_factory(self):
        self.dead_zone = self.factory_dead_zone

    def setup(self, config, kp):
        cfg = (STS_CONFIGS if self.motor == "sts3215" else XL_CONFIGS)[config]
        if config == "F":
            kp = 400                                         # XL330 power-on Position P gain
        self.model.actuator.kp = cfg.get("p", kp)
        self.dead_zone = cfg.get("dead_zone", self.factory_dead_zone)

    def set_goal(self, q):
        self.goal = float(q)

    def set_torque(self, on):
        self.torque = bool(on)

    def read(self):
        self._advance(self.bus_dt)
        q = self.sim.q + self.rng.normal(0, self.noise_ticks) * self.TICK
        return {"position": float(np.round(q / self.TICK) * self.TICK), "speed": float(self.sim.dq),
                "load": 0.0, "input_volts": float(self.model.actuator.vin), "temp": 30.0}

    def _advance(self, s):
        n = max(1, int(round(s / self.sub_dt)))
        dz = self.dead_zone * self.TICK
        for _ in range(n):
            ctrl = self.model.actuator.compute_control(self.goal, self.sim.q, self.sim.dq, self.sub_dt)
            if dz > 0 and abs(self.goal - self.sim.q) <= dz:
                ctrl = 0.0 * ctrl                              # firmware stops driving inside the dead zone
            self.sim.step(ctrl, self.torque, self.sub_dt)
        self.t += n * self.sub_dt
