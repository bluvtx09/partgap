"""Download action/state (all joints) from picked SO-100 datasets. -> data/<author>__<name>.npz"""
import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq

H = {"User-Agent": "partgap-research"}
MAXEP = 24
os.makedirs("data", exist_ok=True)


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=60) as r:
        return r.read()


def names_of(feat):
    n = feat["names"]
    return n["motors"] if isinstance(n, dict) else n


def one(d):
    out = f"data/{d['id'].replace('/', '__')}.npz"
    if os.path.exists(out) or os.path.exists(out + ".fail"):
        return
    try:
        base = f"https://huggingface.co/datasets/{d['id']}/resolve/main/"
        info = json.loads(get(base + "meta/info.json"))
        names = names_of(info["features"]["action"])
        snames = names_of(info["features"]["observation.state"])
        tpl = info["data_path"]
        chunks = info.get("chunks_size", 1000)
        A, S, E = [], [], []
        for ep in range(min(MAXEP, info["total_episodes"])):
            path = tpl.format(episode_chunk=ep // chunks, episode_index=ep)
            t = pq.read_table(io.BytesIO(get(base + path)), columns=["action", "observation.state"])
            A.append(np.array(t.column("action").to_pylist(), dtype=float))
            S.append(np.array(t.column("observation.state").to_pylist(), dtype=float))
            E.append(np.full(len(A[-1]), ep))
        np.savez_compressed(out, action=np.concatenate(A), state=np.concatenate(S), episode=np.concatenate(E),
                            names=np.array(names), snames=np.array(snames), fps=info["fps"])
    except Exception as e:
        open(out + ".fail", "w").write(repr(e)[:300])


if __name__ == "__main__":
    picked = json.load(open("picked.json"))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, picked))
    ok = len([f for f in os.listdir("data") if f.endswith(".npz")])
    print("ok", ok, "fail", len([f for f in os.listdir("data") if f.endswith(".fail")]), flush=True)
