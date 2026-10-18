import pandas as pd
import pytest

from cms.analyze.report import matched_pair, metric_pair, restrict_subset


def test_each_rewrite_uses_its_own_population():
    rows = [
        {"uid": uid, "condition": condition, "mode": mode, "score": score}
        for uid, mode, conditions in (
            ("honest", "honest", ("clean", "naive", "aware")),
            ("naive_only", "attack", ("clean", "naive")),
            ("aware_only", "attack", ("clean", "aware")),
        )
        for score, condition in enumerate(conditions)
    ]
    df = pd.DataFrame(rows)
    naive = matched_pair(df, "naive")
    aware = matched_pair(df, "aware")
    assert set(naive.uid) == {"honest", "naive_only"}
    assert set(aware.uid) == {"honest", "aware_only"}
    rd = restrict_subset(naive, "rd", {"aware_only"})
    assert set(rd.uid) == {"honest"}
    with pytest.raises(RuntimeError, match="duplicate"):
        matched_pair(pd.concat([df, df.iloc[[0]]]), "naive")


def test_discrete_threshold_ties_are_not_suspicious():
    df = pd.DataFrame(
        {
            "mode": ["honest"] * 20 + ["attack"] * 2,
            "score": [0.0] * 20 + [0.0, 1.0],
        }
    )
    tpr, auc = metric_pair(df)
    assert tpr == 0.5
    assert auc == 0.75
