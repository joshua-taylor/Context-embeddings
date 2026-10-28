"""STS-B test + SICK test -> data/tests.json (lower-cased, tokenised word lists).

  python scripts/prepare_tests.py --stsb stsb-test.csv --sick sick-test.txt --out data/tests.json
stsb-test.csv: sentence1, sentence2, score (0-5) per row;  sick-test.txt: the SICK release (tab-separated, header row).
"""
import argparse, csv, json, re
TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
tk = lambda s: TOK.findall(s.lower())
p = argparse.ArgumentParser(); p.add_argument("--stsb", required=True); p.add_argument("--sick", required=True); p.add_argument("--out", required=True); a = p.parse_args()
sts = [(tk(r[0]), tk(r[1]), float(r[2])) for r in csv.reader(open(a.stsb)) if len(r) >= 3]
sick = [(tk(r[1]), tk(r[2]), float(r[3]), r[4]) for i, r in enumerate(csv.reader(open(a.sick), delimiter="\t")) if i and len(r) >= 5]
json.dump(dict(sts=sts, sick=sick), open(a.out, "w")); print(len(sts), "STS-B pairs,", len(sick), "SICK pairs")
