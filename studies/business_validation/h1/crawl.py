"""List LeRobot SO-100 datasets (robot_type 'so100', degrees), one per uploader."""
import json, re, urllib.request, collections

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "partgap-research"}), timeout=60) as r:
        return r.read(), r.headers.get("Link")

seen, rows = set(), []
for term in ["so100", "so-100", "so_100"]:
    url = f"https://huggingface.co/api/datasets?search={term}&limit=1000&full=true"
    pages = 0
    while url and pages < 30:
        body, link = get(url)
        for d in json.loads(body):
            if d["id"] in seen:
                continue
            seen.add(d["id"])
            desc = d.get("description") or ""
            rt = re.search(r'"robot_type":\s*"([^"]+)"', desc)
            cv = re.search(r'"codebase_version":\s*"([^"]+)"', desc)
            ep = re.search(r'"total_episodes":\s*(\d+)', desc)
            fr = re.search(r'"total_frames":\s*(\d+)', desc)
            rows.append({"id": d["id"], "author": d["id"].split("/")[0], "robot_type": rt.group(1) if rt else None,
                         "codebase": cv.group(1) if cv else None, "episodes": int(ep.group(1)) if ep else 0,
                         "frames": int(fr.group(1)) if fr else 0, "downloads": d.get("downloads", 0),
                         "created": d.get("createdAt")})
        m = re.search(r'<([^>]+)>;\s*rel="next"', link or "")
        url = m.group(1) if m else None
        pages += 1
json.dump(rows, open("all_datasets.json", "w"))
c = collections.Counter((r["robot_type"], r["codebase"]) for r in rows)
print(len(rows), "datasets"); print(c.most_common(12))
so = [r for r in rows if r["robot_type"] == "so100" and (r["codebase"] or "").startswith("v2") and r["episodes"] >= 10]
by = collections.defaultdict(list)
for r in so:
    by[r["author"]].append(r)
pick = [max(v, key=lambda r: (r["frames"])) for v in by.values()]
json.dump(pick, open("picked.json", "w"), indent=0)
print("so100 v2 >=10 episodes:", len(so), "authors:", len(pick))
