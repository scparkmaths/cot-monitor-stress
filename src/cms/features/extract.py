"""Extract fixed numeric shape features for every condition row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cms.features.shape import ALL_FEATURE_NAMES, ShapeExtractor
from cms.io import read_jsonl, sha256_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/conditions.jsonl")
    ap.add_argument("--output", default="data/raw/features.parquet")
    ap.add_argument("--cache-dir", default="cache/features")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nll-chunk-tokens", type=int, default=256)
    ap.add_argument("--allow-unpinned-ref-lm", action="store_true")
    ap.add_argument(
        "--skip-nll",
        action="store_true",
        help="smoke/debug only; writes zero NLL features",
    )
    a = ap.parse_args()

    rows = read_jsonl(a.input)
    ext = ShapeExtractor(
        device=a.device,
        allow_unpinned_ref_lm=a.allow_unpinned_ref_lm,
        nll_chunk_tokens=a.nll_chunk_tokens,
    )
    cache_dir = Path(a.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for row in tqdm(rows, desc="shape features"):
        cot = str(row.get("cot", ""))
        # Version tag ensures feature-definition changes don't reuse old cache entries.
        key = sha256_text("shape-v1\0" + cot + ("\0skip-nll" if a.skip_nll else ""))
        cache = cache_dir / key[:2] / f"{key}.json"
        if cache.exists():
            feats = json.loads(cache.read_text(encoding="utf-8"))
        else:
            feats = ext.extract(cot, compute_nll=not a.skip_nll)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(feats, sort_keys=True), encoding="utf-8")
        rec = {
            "uid": row.get("uid"),
            "id": row.get("id"),
            "agent": row.get("agent"),
            "mode": row.get("mode"),
            "main_task": row.get("main_task"),
            "side_task": row.get("side_task"),
            "fold_main": row.get("fold_main"),
            "fold_side": row.get("fold_side"),
            "condition": row.get("condition"),
            "scope": row.get("scope"),
            **{name: float(feats[name]) for name in ALL_FEATURE_NAMES},
        }
        out.append(rec)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(a.output, index=False)
    print(f"wrote {len(out)} rows -> {a.output}")


if __name__ == "__main__":
    main()
