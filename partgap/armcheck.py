"""How does your SO-100/101 arm compare? Stopping hysteresis of shoulder_pan vs 193 public arms.

    pip install "partgap[armcheck] @ git+https://github.com/bluvtx09/partgap"
    partgap-armcheck <hf_user>/<dataset>          # a LeRobot dataset you recorded (private: set HF_TOKEN)
    partgap-armcheck --local path/to/dataset       # a dataset folder on disk

Hysteresis = (position error after the joint stopped coming down) - (after it stopped going up), averaged
over every pause in your episodes. It is how far the servo stops short, depending on the direction.
The method is the same as the one behind the reference numbers (PartGap, studies/business_validation),
including the requirement of 5 pauses per direction in both the even and the odd episodes.
The reference holds SO-100 datasets (LeRobot v2.x, angles in degrees). SO-101 uses the same STS3215 servo, but
SO-101 and v3.x datasets were not part of the reference. Needs angles in degrees (LeRobot `use_degrees=True`).
"""
import argparse
import io
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOL, MOVE = 0.3, 2.0
MAX_EPISODES = 24


def rests(a, s, fps):
    """(direction, error) for each pause: command range <= 0.3 deg for >= 0.5 s after a >= 2 deg move."""
    L = int(round(0.5 * fps))
    i, n = fps, len(a)
    while i < n - L:
        j = i + 1
        lo = hi = a[i]
        while j < n:
            lo, hi = min(lo, a[j]), max(hi, a[j])
            if hi - lo > TOL:
                break
            j += 1
        if j - i >= L:
            move = a[i] - a[i - fps]
            if abs(move) >= MOVE:
                k = i + (j - i) // 2
                yield (1 if move > 0 else -1), float(np.mean(s[k:j] - a[k:j]))
            i = j
        else:
            i += 1


def hysteresis(items):
    pos = [e for d, e in items if d > 0]
    neg = [e for d, e in items if d < 0]
    if len(pos) < 5 or len(neg) < 5:
        return None, len(pos), len(neg)
    return float(np.mean(neg) - np.mean(pos)), len(pos), len(neg)


# ------------------------------------------------------------------ reading LeRobot datasets
def _get(url):
    headers = {"User-Agent": "partgap-armcheck"}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return r.read()


def _names(feat):
    n = feat["names"]
    return n["motors"] if isinstance(n, dict) else n


def _read_table(raw):
    import pyarrow.parquet as pq
    t = pq.read_table(io.BytesIO(raw), columns=["action", "observation.state", "episode_index"])
    return (np.array(t.column("action").to_pylist(), float), np.array(t.column("observation.state").to_pylist(), float),
            np.array(t.column("episode_index").to_pylist(), int))


def load_dataset(repo=None, local=None):
    """-> info, action, state, episode (first MAX_EPISODES episodes)."""
    if local:
        rd = lambda p: open(os.path.join(local, p), "rb").read()  # noqa: E731
    else:
        base = f"https://huggingface.co/datasets/{repo}/resolve/main/"
        rd = lambda p: _get(base + p)  # noqa: E731
    info = json.loads(rd("meta/info.json"))
    files = []
    if "{episode_index" in info["data_path"]:                     # v2.x: one file per episode
        cs = info.get("chunks_size", 1000)
        for ep in range(min(MAX_EPISODES, info["total_episodes"])):
            files.append(info["data_path"].format(episode_chunk=ep // cs, episode_index=ep))
    else:                                                         # v3.x: chunked files
        if local:
            for root, _, fs in os.walk(os.path.join(local, "data")):
                files += [os.path.relpath(os.path.join(root, f), local) for f in fs if f.endswith(".parquet")]
        else:
            tree = json.loads(_get(f"https://huggingface.co/api/datasets/{repo}/tree/main/data?recursive=true"))
            files = [f["path"] for f in tree if f["path"].endswith(".parquet")]
        files.sort()
    A, S, E = [], [], []
    for f in files:
        a, s, e = _read_table(rd(f))
        A.append(a), S.append(s), E.append(e)
        if len(np.unique(np.concatenate(E))) >= MAX_EPISODES:
            break
    a, s, e = np.concatenate(A), np.concatenate(S), np.concatenate(E)
    keep = np.isin(e, np.unique(e)[:MAX_EPISODES])
    return info, a[keep], s[keep], e[keep]


def check(info, a_all, s_all, ep, degrees_confirmed=False):
    names = _names(info["features"]["action"])
    snames = _names(info["features"]["observation.state"])
    ja = next((i for i, x in enumerate(names) if x.split(".")[0].endswith("shoulder_pan")), None)
    js = next((i for i, x in enumerate(snames) if x.split(".")[0].endswith("shoulder_pan")), None)
    if ja is None or js is None:
        raise SystemExit("No shoulder_pan joint in this dataset (SO-100/101 arms only).")
    a, s = a_all[:, ja], s_all[:, js]
    if np.nanmax(np.abs(a)) > 400:
        raise SystemExit("Angles look like raw encoder ticks, not degrees. Record with degrees to use this check.")
    if info.get("robot_type") != "so100" and np.nanmax(np.abs(a)) <= 100.5 and not degrees_confirmed:
        raise SystemExit("All shoulder_pan values are within +-100. LeRobot's normalized mode (use_degrees=False) also "
                         "gives that range, and then this check means nothing.\nIf you recorded in degrees, run again "
                         "with --degrees.")
    fps = int(info["fps"])
    halves = {0: [], 1: []}
    for e in np.unique(ep):
        m = ep == e
        halves[int(e) % 2] += list(rests(a[m], s[m], fps))
    h_even, h_odd = hysteresis(halves[0]), hysteresis(halves[1])
    h = hysteresis(halves[0] + halves[1])
    if h_even[0] is None or h_odd[0] is None:                  # same rule as the reference arms
        return None, h[1], h[2]
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo", nargs="?", help="<hf_user>/<dataset>")
    ap.add_argument("--local", help="dataset folder on disk instead of the Hub")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--degrees", action="store_true", help="confirm the dataset angles are in degrees")
    a = ap.parse_args()
    if not a.repo and not a.local:
        ap.error("give a Hub dataset id or --local")
    info, act, st, ep = load_dataset(a.repo, a.local)
    h, n_pos, n_neg = check(info, act, st, ep, a.degrees)
    ref = json.load(open(os.path.join(HERE, "armcheck_field.json")))
    H = np.array(ref["H"])
    out = {"dataset": a.repo or a.local, "robot_type": info.get("robot_type"), "episodes_used": int(len(np.unique(ep))),
           "pauses_up": n_pos, "pauses_down": n_neg, "hysteresis_deg": h}
    if h is not None:
        out["percentile_among_193_arms"] = float((H < h).mean() * 100)
        out["reference_median_deg"] = float(np.median(H))
    if a.json:
        print(json.dumps(out))
        return
    if h is None:
        print(f"Not enough pauses to measure ({n_pos} after moving up, {n_neg} after moving down; need 5 of each "
              "in both the even and the odd episodes).")
        print("Episodes where the arm stops and holds still for half a second or more work best.")
        sys.exit(1)
    print(f"shoulder_pan stopping hysteresis: {h:.2f} deg  ({n_pos} + {n_neg} pauses, {out['episodes_used']} episodes)")
    print(f"That is larger than {out['percentile_among_193_arms']:.0f}% of 193 public SO-100 arms "
          f"(median {out['reference_median_deg']:.2f} deg, middle half {np.percentile(H, 25):.2f}-{np.percentile(H, 75):.2f}).")
    if info.get("robot_type") != "so100":
        print(f"Note: the reference is SO-100 datasets only; yours is '{info.get('robot_type')}'. Same servo, "
              "but the comparison is not validated for it.")


if __name__ == "__main__":
    main()
