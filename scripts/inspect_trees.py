import json,sys,os,glob
root=sys.argv[1]
def walk(n):
    yield n
    for c in n.get("children",[]): yield from walk(c)
for f in sorted(glob.glob(f"{root}/*/*/*ha_12y.json")+glob.glob(f"{root}/*/*/*ha_12y_tree.json")):
    j=json.load(open(f))
    if "tree" in j:  # v2
        nodes=list(walk(j["tree"])); tips=[n for n in nodes if not n.get("children")]
        dates=[n["node_attrs"]["num_date"]["value"] for n in tips if "num_date" in n.get("node_attrs",{})]
        upd=j["meta"].get("updated"); title=j["meta"].get("title")
        ver=j.get("version")
    else:  # v1 tree.json
        nodes=list(walk(j)); tips=[n for n in nodes if not n.get("children")]
        dates=[n["attr"]["num_date"] for n in tips if "num_date" in n.get("attr",{})]
        m=json.load(open(f.replace("_tree.json","_meta.json"))) if os.path.exists(f.replace("_tree.json","_meta.json")) else {}
        upd=m.get("updated"); title=m.get("title"); ver="v1"
    print(f"{os.path.relpath(f,root):60s} {ver!s:5} updated={upd} tips={len(tips):5d} max_date={max(dates):.3f} min_date={min(dates):.2f} | {title}")
