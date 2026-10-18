"""Stack content + structure scores with a per-agent logistic regression."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KEYS = [
    "uid",
    "id",
    "agent",
    "mode",
    "main_task",
    "side_task",
    "condition",
    "scope",
    "fold_main",
    "fold_side",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", default="data/raw/structure_scores.parquet")
    ap.add_argument("--content", default="data/raw/content_scores.parquet")
    ap.add_argument("--output", default="data/raw/combined_scores.parquet")
    a = ap.parse_args()

    s = pd.read_parquet(a.structure)
    s = s[(s.monitor == "structure_lr") & (s.cv == "main")][KEYS + ["score"]].rename(
        columns={"score": "structure"}
    )
    c = pd.read_parquet(a.content)
    out = []
    for content_name, combined_name in (("content_mid", "combined_mid_lr"),):
        cc = c[c.monitor == content_name][KEYS + ["score"]].rename(
            columns={"score": "content"}
        )
        df = s.merge(cc, on=KEYS, how="inner", validate="one_to_one")
        base = df[(df.condition == "clean") & (df.scope == "attack-only")]
        for agent in sorted(df.agent.unique()):
            ga = df[df.agent == agent]
            ba = base[base.agent == agent]
            for fold in sorted(int(x) for x in ga.fold_main.unique()):
                train = ba[ba.fold_main != fold]
                test = ga[ga.fold_main == fold]
                pipe = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                max_iter=5000, class_weight="balanced", random_state=0
                            ),
                        ),
                    ]
                )
                pipe.fit(
                    train[["content", "structure"]],
                    (train["mode"] == "attack").astype(int),
                )
                scores = pipe.predict_proba(test[["content", "structure"]])[:, 1]
                for (_, r), score in zip(test.iterrows(), scores):
                    out.append(
                        {
                            "uid": r.uid,
                            "id": r.id,
                            "agent": r.agent,
                            "mode": r["mode"],
                            "main_task": r.main_task,
                            "side_task": r.side_task,
                            "condition": r.condition,
                            "scope": r.scope,
                            "monitor": combined_name,
                            "cv": "main",
                            "fold_main": int(r.fold_main),
                            "fold_side": None
                            if pd.isna(r.fold_side)
                            else int(r.fold_side),
                            "score": float(score),
                        }
                    )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(a.output, index=False)
    print(f"wrote {len(out)} scores -> {a.output}")


if __name__ == "__main__":
    main()
