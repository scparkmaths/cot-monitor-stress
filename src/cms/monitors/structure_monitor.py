"""Structure-only LR/RF monitors with leave-one-main-task-out and side-task CV."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cms.features.shape import FEATURE_NAMES, LENGTH_FEATURE_NAMES


def label(df: pd.DataFrame) -> np.ndarray:
    return (df["mode"].to_numpy() == "attack").astype(int)


def lr_pipeline(features: list[str]) -> Pipeline:
    return Pipeline(
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


def rf_pipeline(features: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_score(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], kind: str
) -> np.ndarray:
    if train.empty or len(np.unique(label(train))) < 2:
        raise RuntimeError("training fold has fewer than two classes")
    pipe = lr_pipeline(features) if kind == "lr" else rf_pipeline(features)
    pipe.fit(train[features], label(train))
    return pipe.predict_proba(test[features])[:, 1]


def records(
    test: pd.DataFrame, scores: np.ndarray, monitor: str, cv: str
) -> list[dict]:
    out = []
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
                "monitor": monitor,
                "cv": cv,
                "fold_main": int(r.fold_main),
                "fold_side": None if pd.isna(r.fold_side) else int(r.fold_side),
                "score": float(score),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/raw/features.parquet")
    ap.add_argument("--output", default="data/raw/structure_scores.parquet")
    ap.add_argument("--models-dir", default="outputs/models")
    ap.add_argument(
        "--importances", default="outputs/tables/structure_feature_importances.csv"
    )
    a = ap.parse_args()

    df = pd.read_parquet(a.features)
    base = df[(df.condition == "clean") & (df.scope == "attack-only")].copy()
    shape = list(FEATURE_NAMES)
    shape_len = shape + list(LENGTH_FEATURE_NAMES)
    specs = [
        ("structure_lr", "lr", shape),
        ("structure_rf", "rf", shape),
        ("structure_lr_with_length", "lr", shape_len),
        ("structure_rf_with_length", "rf", shape_len),
    ]
    out: list[dict] = []

    for agent in sorted(df.agent.unique()):
        ga = df[df.agent == agent]
        ba = base[base.agent == agent]
        # Primary leave-one-main-task-out predictions for every condition/scope.
        for fold in sorted(int(x) for x in ga.fold_main.dropna().unique()):
            train = ba[ba.fold_main != fold]
            test = ga[ga.fold_main == fold]
            for name, kind, feats in specs:
                out += records(test, fit_score(train, test, feats, kind), name, "main")

        # Leave-one-side-task-out attack predictions. Honest OOF reference scores
        # remain the main-task-CV scores, avoiding in-sample honest thresholds.
        sides = sorted(
            int(x)
            for x in ba.loc[ba["mode"] == "attack", "fold_side"].dropna().unique()
        )
        for side in sides:
            train = ba[(ba["mode"] == "honest") | (ba.fold_side != side)]
            test = ga[(ga["mode"] == "attack") & (ga.fold_side == side)]
            for name, kind, feats in specs[:2]:
                out += records(
                    test, fit_score(train, test, feats, kind), name + "_sidecv", "side"
                )

        # Final clean fit only for interpretable coefficient/importances artifact.
        Path(a.models_dir).mkdir(parents=True, exist_ok=True)
        for name, kind, feats in specs[:2]:
            pipe = lr_pipeline(feats) if kind == "lr" else rf_pipeline(feats)
            pipe.fit(ba[feats], label(ba))
            joblib.dump(pipe, Path(a.models_dir) / f"{name}_{agent}.joblib")

    odf = pd.DataFrame(out)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    odf.to_parquet(a.output, index=False)

    imp_rows = []
    for agent in sorted(base.agent.unique()):
        for name in ("structure_lr", "structure_rf"):
            pipe = joblib.load(Path(a.models_dir) / f"{name}_{agent}.joblib")
            model = pipe.named_steps["model"]
            vals = model.coef_[0] if name.endswith("lr") else model.feature_importances_
            for feat, value in zip(FEATURE_NAMES, vals):
                imp_rows.append(
                    {
                        "agent": agent,
                        "monitor": name,
                        "feature": feat,
                        "importance": float(value),
                    }
                )
    Path(a.importances).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(imp_rows).to_csv(a.importances, index=False)
    print(f"wrote {len(odf)} scores -> {a.output}")


if __name__ == "__main__":
    main()
