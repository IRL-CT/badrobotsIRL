#!/usr/bin/env python3
"""Uncapped W&B export for v4-v6, with run ids and URLs so every row is traceable.

The UI export stops at 1000 rows, which is why both linear exports in
corrections/ are exactly 1000 rows -- any "best config" drawn from them is
best-of-1000, not best-of-all.

Two things this does that the older export_wandb.py did not:
  * captures the run ID and a direct URL, not just the display name (a name like
    "volcanic-sweep-47" is not addressable; tracing a row needs the id)
  * reads the sweep from the run's cached attrs instead of `run.sweep.id`, which
    lazily fires a separate API request per run and makes a large export crawl

Writes one CSV per project; each project file is written whole when it finishes.
"""
import argparse
import os
import sys
import time

import pandas as pd
import wandb


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def sweep_of(r):
    """Sweep id without triggering a per-run lazy fetch."""
    a = getattr(r, "_attrs", {}) or {}
    for k in ("sweepName", "sweep_name", "sweepId"):
        if a.get(k):
            return a[k]
    try:
        return r.sweep.id if r.sweep else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--projects", nargs="+", required=True)
    ap.add_argument("--out", default="raw")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave already-exported project files alone (resume)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    api = wandb.Api(timeout=120)
    for proj in a.projects:
        p_done = os.path.join(a.out, f"{proj}.csv")
        if a.skip_existing and os.path.exists(p_done) and os.path.getsize(p_done) > 0:
            print(f"{proj}: already exported, skipping", flush=True)
            continue
        t0 = time.time()
        rows = []
        for i, r in enumerate(api.runs(f"{a.entity}/{proj}", per_page=500), 1):
            row = {"project": proj, "Name": r.name, "ID": r.id, "State": r.state,
                   "Sweep": sweep_of(r), "Created": r.created_at,
                   "Runtime": r.summary.get("_runtime"),
                   "URL": f"https://wandb.ai/{a.entity}/{proj}/runs/{r.id}"}
            row.update({k: v for k, v in r.config.items() if not k.startswith("_")})
            try:
                row.update(flatten(r.summary._json_dict))
            except Exception as e:
                row["_summary_error"] = str(e)
            rows.append(row)
            if i % 250 == 0:
                print(f"  {proj}: {i} runs ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        p = os.path.join(a.out, f"{proj}.csv")
        df.to_csv(p, index=False)
        print(f"{proj}: {len(df)} runs, {df.shape[1]} cols -> {p} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
