import csv
import json
import sys
from pathlib import Path

truth = json.loads(Path(sys.argv[1], "truth.json").read_text())
# Always answering "heard" gets 60/80 here, so also report accuracy split by whether the truth is the heard value.
hits = {True: [0, 0], False: [0, 0]}
for r in csv.DictReader(Path(sys.argv[2], "sites.csv").read_text().splitlines()):
    t = truth[f"{r['participant']}_{r['sentence']}_{r['heard']}"][r["site"]]
    h = hits[t == r["value_heard"]]
    h[0] += r["value_estimated"] == t
    h[1] += 1
ok, n = hits[True][0] + hits[False][0], hits[True][1] + hits[False][1]
bal = sum(a / b for a, b in hits.values() if b) / sum(1 for _, b in hits.values() if b)
print(
    f"{ok}/{n}  truth=heard {hits[True][0]}/{hits[True][1]}  truth=other {hits[False][0]}/{hits[False][1]}  balanced {bal:.2f}"
)
