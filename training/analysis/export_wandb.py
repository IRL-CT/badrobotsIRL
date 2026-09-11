#!/usr/bin/env python3
"""Uncapped W&B export (the UI export stops at 1000 rows).

Usage:  python3 export_wandb.py --entity ENT --projects proj1 proj2 ... --out exports/
Writes one CSV per project with config + summary flattened, matching the UI export column names
as closely as possible (Name, State, Sweep, Created + all config keys + all summary keys).
"""
import argparse, os, pandas as pd, wandb

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--projects", nargs="+", required=True)
    ap.add_argument("--out", default="exports")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    api = wandb.Api(timeout=120)
    for proj in a.projects:
        rows = []
        for r in api.runs(f"{a.entity}/{proj}", per_page=500):
            row = {"Name": r.name, "State": r.state, "Sweep": r.sweep.id if r.sweep else None,
                   "Created": r.created_at, "Runtime": r.summary.get("_runtime")}
            row.update({k: v for k, v in r.config.items() if not k.startswith("_")})
            row.update(flatten(r.summary._json_dict))
            rows.append(row)
        df = pd.DataFrame(rows)
        p = os.path.join(a.out, f"{proj}.csv"); df.to_csv(p, index=False)
        print(f"{proj}: {len(df)} runs -> {p}")

if __name__ == "__main__":
    main()
