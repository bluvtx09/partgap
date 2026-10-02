import numpy as np

from partgap.armcheck import hysteresis, rests


def test_hysteresis_recovers_planted_offset():
    """A servo that stops 0.3 deg short of every target: error -0.3 after moving up, +0.3 after moving down."""
    fps, hold, move = 30, 60, 15
    a, s = [], []
    target = 0.0
    for k in range(24):
        new = [10.0, 0.0, -10.0, 0.0][k % 4]
        ramp = np.linspace(target, new, move)
        short = -0.3 if new > target else 0.3
        a += list(ramp) + [new] * hold
        s += list(ramp) + [new + short] * hold
        target = new
    h, n_pos, n_neg = hysteresis(list(rests(np.array(a), np.array(s), fps)))
    assert n_pos >= 5 and n_neg >= 5
    # the pause window ends on the first sample of the next move (where state == command), as in the field
    # analysis, so the estimate is slightly diluted
    assert abs(h - 0.6) < 0.03


def test_too_few_pauses_returns_none():
    a = np.zeros(300)
    h, n_pos, n_neg = hysteresis(list(rests(a, a, 30)))
    assert h is None and n_pos == 0 and n_neg == 0
