"""Assemble component monitor outputs into data/raw/scores.parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/raw/scores.parquet")
    ap.add_argument("--structure", default="data/raw/structure_scores.parquet")
    ap.add_argument("--length", default="data/raw/length_scores.parquet")
    ap.add_argument("--content", default="data/raw/content_scores.parquet")
    ap.add_argument("--combined", default="data/raw/combined_scores.parquet")
    ap.add_argument("--actions", default="data/raw/actions_only_scores.parquet")
    a = ap.parse_args()

    frames = [
        pd.read_parquet(p) for p in (a.structure, a.length, a.content, a.combined)
    ]
    apath = Path(a.actions)
    if apath.exists():
        x = pd.read_parquet(apath).copy()
        x["condition"] = "clean"
        x["scope"] = "attack-only"
        x["monitor"] = "actions_only"
        x["cv"] = "fixed"
        frames.append(x)
    df = pd.concat(frames, ignore_index=True, sort=False)
    # Canonical column order; extra provenance columns remain after these.
    first = [
        "uid",
        "id",
        "agent",
        "mode",
        "condition",
        "scope",
        "monitor",
        "cv",
        "fold_main",
        "fold_side",
        "score",
    ]
    cols = [c for c in first if c in df.columns] + [
        c for c in df.columns if c not in first
    ]
    df = df[cols]
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.output, index=False)
    print(
        df.groupby(["monitor", "condition", "scope"], dropna=False).size().to_string()
    )
    print(f"\nwrote {len(df)} score rows -> {a.output}")


if __name__ == "__main__":
    main()
