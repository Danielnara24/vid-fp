import json, os, sys, collections
import order as O, score as S
def scan(path):
    out = []
    for line in open(path):
        rec = json.loads(line)
        a, b = os.path.relpath(rec["a"], S.VIDS), os.path.relpath(rec["b"], S.VIDS)
        if a in O.man and b in O.man and O.label(a, b) == "FP_other":
            out.append((a, b, rec))
    return out
d = sys.argv[1]
for k in range(1, 13):
    r = scan(f"{d}/sweep-d{k}.jsonl")
    kinds = collections.Counter()
    for a, b, rec in r:
        ta, tb = O.man[a]["transform"], O.man[b]["transform"]
        o = "mir" if rec["mir"] else ("ref" if rec["ref"] else "plain")
        kinds[o] += 1
    seeds = collections.Counter(tuple(sorted((O.man[a]["seed"], O.man[b]["seed"]))) for a, b, _ in r)
    print(f"d{k:2} unrelated {len(r):5}  {dict(kinds)}  top {seeds.most_common(4)}")
