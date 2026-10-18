"""Length-only LR baseline under leave-one-main-task-out CV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["n_chars", "n_sents"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/raw/features.parquet")
    ap.add_argument("--output", default="data/raw/length_scores.parquet")
    a = ap.parse_args()
    df = pd.read_parquet(a.features)
    base = df[(df.condition == "clean") & (df.scope == "attack-only")]
    out = []
    for agent in sorted(df.agent.unique()):
        ga = df[df.agent == agent]
        ba = base[base.agent == agent]
        for fold in sorted(int(x) for x in ga.fold_main.dropna().unique()):
            train = ba[ba.fold_main != fold]
            test = ga[ga.fold_main == fold]
            pipe = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=5000, class_weight="balanced", random_state=0
                        ),
                    ),
                ]
            )
            pipe.fit(train[FEATURES], (train["mode"] == "attack").astype(int))
            scores = pipe.predict_proba(test[FEATURES])[:, 1]
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
                        "monitor": "length_lr",
                        "cv": "main",
                        "fold_main": int(r.fold_main),
                        "fold_side": None if pd.isna(r.fold_side) else int(r.fold_side),
                        "score": float(score),
                    }
                )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(a.output, index=False)
    print(f"wrote {len(out)} scores -> {a.output}")


if __name__ == "__main__":
    main()
