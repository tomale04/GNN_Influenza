"""Download Nextstrain seasonal-flu HA 12y trees (h3n2, h1n1pdm) as of April 1 each year.

Source: versioned public S3 bucket nextstrain-data (backs data.nextstrain.org).
For each year/subtype, picks the latest version of the main dataset JSON with
LastModified <= YYYY-04-01T23:59:59Z, plus sidecar files from the same build.
Usage: python3 -I download_trees.py OUT_DIR
"""
import sys, os, json, urllib.request, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_versions import list_versions

BASE = "https://nextstrain-data.s3.amazonaws.com/"
# (prefix, main-file suffix, sidecar suffixes) per naming era
ERAS = [
    ("flu_seasonal_{s}_ha_12y", "_tree.json", ["_meta.json", "_tip-frequencies.json", "_root-sequence.json", "_frequencies.json", "_entropy.json", "_sequences.json"]),  # auspice v1, 2018-07..2019
    ("flu_seasonal_{s}_ha_12y", ".json", ["_tip-frequencies.json", "_root-sequence.json"]),  # auspice v2, 2019..2024-04
    ("seasonal-flu_{s}_ha_12y", ".json", ["_tip-frequencies.json", "_root-sequence.json"]),  # 2024-04..present
]
SUBTYPES = ["h3n2", "h1n1pdm"]
YEARS = range(2017, 2027)
SIDECAR_WINDOW_S = 3 * 3600  # sidecars must be uploaded within 3h before/after main file

def ts(s):
    from datetime import datetime
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()

def main(out):
    os.makedirs(out, exist_ok=True)
    cache = {}
    def versions(prefix):
        if prefix not in cache:
            cache[prefix] = [v for v in list_versions(prefix) if not v["delete_marker"]]
        return cache[prefix]
    manifest = []
    for s in SUBTYPES:
        for y in YEARS:
            cutoff = f"{y}-04-01T23:59:59.999Z"
            cands = []
            for pfx, main_sfx, sidecars in ERAS:
                p = pfx.format(s=s)
                cands += [(v, p, sidecars) for v in versions(p) if v["key"] == p + main_sfx and v["last_modified"] <= cutoff]
            if not cands:
                print(f"{y} {s}: none available"); continue
            v, p, sidecars = max(cands, key=lambda c: c[0]["last_modified"])
            if v["last_modified"][:4] != str(y):
                print(f"{y} {s}: latest <= cutoff is {v['last_modified']} (prior year) -> skipping"); continue
            d = os.path.join(out, s, str(y)); os.makedirs(d, exist_ok=True)
            files = [v]
            for sc in sidecars:
                near = [w for w in versions(p) if w["key"] == p + sc and abs(ts(w["last_modified"]) - ts(v["last_modified"])) <= SIDECAR_WINDOW_S and w["last_modified"] <= cutoff]
                if near: files.append(min(near, key=lambda w: abs(ts(w["last_modified"]) - ts(v["last_modified"]))))
                elif sc == "_root-sequence.json":
                    # root-sequence was not re-uploaded every build when unchanged; fall back to the
                    # latest version uploaded before the main file. Validated downstream by checking
                    # mutation from-states against the reconstructed sequences.
                    prior = [w for w in versions(p) if w["key"] == p + sc and w["last_modified"] <= v["last_modified"]]
                    if prior:
                        w = max(prior, key=lambda w: w["last_modified"]); files.append({**w, "fallback_prior_version": True})
                        print(f"  {y} {s}: root-sequence fallback to prior version {w['last_modified']}")
            rec = {"subtype": s, "year": y, "cutoff": cutoff, "files": []}
            for f in files:
                url = f"{BASE}{f['key']}?versionId={f['version_id']}"
                req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
                data = urllib.request.urlopen(req, timeout=300).read()
                if data[:2] == b"\x1f\x8b":
                    import gzip; data = gzip.decompress(data)
                json.loads(data)  # validate
                dest = os.path.join(d, f["key"]); open(dest, "wb").write(data)
                rec["files"].append({**f, "url": url, "local_path": os.path.relpath(dest, out), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
            manifest.append(rec)
            print(f"{y} {s}: {v['key']} @ {v['last_modified']} (+{len(files)-1} sidecars)")
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)

if __name__ == "__main__":
    main(sys.argv[1])
