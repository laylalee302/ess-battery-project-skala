"""후보 모델 비교 -> 최종 모델. 후보와 선택 규칙은 DAY 1 보고서 8장(Modeling Strategy)을 따른다.

누수 방지:
  - 스케일러와 2차 항 생성은 Pipeline 안에 있어 학습 fold 데이터로만 fit된다.
  - 하이퍼파라미터는 GridSearchCV + GroupKFold(그룹 = 프로토콜)로 고른다. 같은 프로토콜 셀이 학습/검증으로
    갈라지면 튜닝이 새므로 RidgeCV(LOO), ElasticNetCV(일반 K-fold)는 쓰지 않는다.
  - 이 모듈은 Batch 2 를 읽지 않는다. 호출하는 쪽(run_pipeline)이 Batch 1 표만 넘긴다.
Target 은 log10(cycle_life) 이며 MAPE 는 10**로 되돌려 계산한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import SEED
from .evaluate import mape
from .features import FEATURE_SETS, TARGET, select_features
from .split import group_kfold

INNER_FOLDS = 3
OUTER_FOLDS = 5
GROUP_COL = "policy"


def _log_mape(y_log_true, y_log_pred) -> float:
    return mape(10 ** np.asarray(y_log_true), 10 ** np.asarray(y_log_pred))


_MAPE_SCORER = make_scorer(_log_mape, greater_is_better=False)


def _add_square(X):
    """첫 번째 열(dq_log10_var)의 제곱 항을 오른쪽에 붙인다."""
    X = np.asarray(X, dtype=float)
    return np.column_stack([X, X[:, 0] ** 2])


def _scaled(model, quad: bool = False) -> Pipeline:
    steps = []
    if quad:
        steps.append(("square", FunctionTransformer(_add_square)))
    steps += [("scale", StandardScaler()), ("model", model)]
    return Pipeline(steps)


# 이름 -> (복잡도 순위, 허용 피처 세트, 기본 파이프라인, 튜닝 그리드). 순위가 낮을수록 단순한 모델.
CANDIDATES: dict[str, dict] = {
    "baseline_median": {"rank": 0, "sets": ["A"], "pipe": lambda: Pipeline([("model", DummyRegressor(strategy="median"))]), "grid": {}},
    "ols": {"rank": 1, "sets": ["A"], "pipe": lambda: _scaled(LinearRegression()), "grid": {}},
    "ridge": {"rank": 2, "sets": ["A", "B"], "pipe": lambda: _scaled(Ridge()),
              "grid": {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}},
    "elastic_net": {"rank": 3, "sets": ["A", "B"], "pipe": lambda: _scaled(ElasticNet(max_iter=20000, random_state=SEED)),
                    "grid": {"model__alpha": [0.0001, 0.001, 0.01, 0.1], "model__l1_ratio": [0.2, 0.5, 0.8]}},
    "ridge_quad": {"rank": 4, "sets": ["A"], "pipe": lambda: _scaled(Ridge(), quad=True),
                   "grid": {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}},
    "huber": {"rank": 5, "sets": ["A", "B"], "pipe": lambda: _scaled(HuberRegressor(max_iter=2000)),
              "grid": {"model__epsilon": [1.35, 1.75], "model__alpha": [0.0001, 0.01, 1]}},
    "random_forest": {"rank": 6, "sets": ["A", "B"],
                      "pipe": lambda: Pipeline([("model", RandomForestRegressor(n_estimators=300, random_state=SEED))]),
                      "grid": {"model__max_depth": [2, 4], "model__min_samples_leaf": [1, 3]}},
    "gradient_boosting": {"rank": 7, "sets": ["A", "B"],
                          "pipe": lambda: Pipeline([("model", GradientBoostingRegressor(random_state=SEED))]),
                          "grid": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
                                   "model__max_depth": [2, 3]}},
}
REFERENCE_ONLY = {"baseline_median"}   # 참고 기준선: 최종 선택 대상에서 제외


def _assert_batch1(table: pd.DataFrame) -> None:
    """모델 선택/튜닝/최종 학습에는 Batch 1 셀만 쓴다 (Batch 2 격리)."""
    if "batch" in table.columns and set(table["batch"]) != {"batch1"}:
        raise AssertionError(f"Batch 1 이외의 셀이 학습 단계에 들어옴: {sorted(set(table['batch']))}")


def make_estimator(name: str, inner_folds: int = INNER_FOLDS):
    """튜닝이 필요한 후보는 GridSearchCV(GroupKFold)로 감싼다. fit(X, y, groups=...) 로 호출한다."""
    spec = CANDIDATES[name]
    pipe = spec["pipe"]()
    if not spec["grid"]:
        return pipe
    return GridSearchCV(pipe, spec["grid"], cv=group_kfold(inner_folds), scoring=_MAPE_SCORER, refit=True)


def _fit(est, X, y, groups):
    est = clone(est)
    if isinstance(est, GridSearchCV):
        est.fit(X, y, groups=groups)
    else:
        est.fit(X, y)
    return est


def cross_val_predict_groups(name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                             n_splits: int = OUTER_FOLDS) -> tuple[pd.Series, list[float]]:
    """중첩 CV. 바깥 fold 는 프로토콜 단위로 나누고, 각 fold 의 학습 데이터 안에서 하이퍼파라미터를 고른다.
    반환: (셀별 out-of-fold 예측(log10), fold 별 MAPE 리스트)."""
    oof = pd.Series(np.nan, index=X.index)
    fold_mapes = []
    for tr, va in group_kfold(n_splits).split(X, y, groups):
        est = _fit(make_estimator(name), X.iloc[tr], y.iloc[tr], groups.iloc[tr])
        oof.iloc[va] = est.predict(X.iloc[va])
        fold_mapes.append(_log_mape(y.iloc[va], oof.iloc[va]))
    return oof, fold_mapes


def evaluate_candidates(table: pd.DataFrame, train_ids, valid_ids) -> tuple[pd.DataFrame, dict]:
    """모든 (후보, 피처 세트) 조합을 Train CV 와 Valid 로 평가한다. Batch 1 표만 받는다.

    반환: (비교표, {(name, set): 셀별 held-out 예측(log10; Train 은 CV 예측, Valid 는 Train 으로 학습한 모델의 예측)}).
    """
    _assert_batch1(table)
    tr, va = table.loc[train_ids], table.loc[valid_ids]
    y_tr, y_va = tr[TARGET], va[TARGET]
    rows, preds = [], {}
    for name, spec in CANDIDATES.items():
        for fs in spec["sets"]:
            X_tr, X_va = select_features(tr, fs), select_features(va, fs)
            oof, fold_mapes = cross_val_predict_groups(name, X_tr, y_tr, tr[GROUP_COL])
            est = _fit(make_estimator(name), X_tr, y_tr, tr[GROUP_COL])
            p_va = pd.Series(est.predict(X_va), index=X_va.index)
            held_pred = pd.concat([oof, p_va])
            held_true = pd.concat([y_tr, y_va])
            ape = np.abs(10 ** held_true - 10 ** held_pred) / 10 ** held_true * 100
            rows.append({
                "model": name, "feature_set": fs, "n_features": len(FEATURE_SETS[fs]) + (1 if name == "ridge_quad" else 0),
                "cv_mape_mean": float(np.mean(fold_mapes)), "cv_mape_std": float(np.std(fold_mapes, ddof=1)),
                "valid_mape": _log_mape(y_va, p_va),
                "heldout_mape": float(ape.mean()), "heldout_se": float(ape.std(ddof=1) / np.sqrt(len(ape))),
                "complexity": spec["rank"] * 2 + (fs == "B"),
                "reference_only": name in REFERENCE_ONLY,
            })
            preds[(name, fs)] = held_pred
    return pd.DataFrame(rows), preds


def select_model(comparison: pd.DataFrame) -> pd.Series:
    """선택 규칙(보고서 8장): Batch 1 held-out(Train CV 36셀 + Valid 10셀) MAPE 가 가장 낮은 후보.
    최저값 + 1 표준오차 이내에 단순한 후보가 있으면 그중 가장 단순한 모델(복잡도 순위, 피처 세트 A 우선)을 고른다."""
    pool = comparison[~comparison["reference_only"]]
    best = pool.loc[pool["heldout_mape"].idxmin()]
    near = pool[pool["heldout_mape"] <= best["heldout_mape"] + best["heldout_se"]]
    return near.sort_values(["complexity", "heldout_mape"]).iloc[0]


def fit_final(name: str, feature_set: str, table: pd.DataFrame):
    """선택된 후보를 Batch 1 전체 셀로 다시 학습한다 (하이퍼파라미터는 GroupKFold(3)으로 다시 고른다)."""
    _assert_batch1(table)
    X, y = select_features(table, feature_set), table[TARGET]
    return _fit(make_estimator(name), X, y, table[GROUP_COL])


def cv_predict_excluding(name: str, feature_set: str, table: pd.DataFrame, exclude_ids=(),
                         n_splits: int = OUTER_FOLDS) -> pd.Series:
    """Batch 1 전체 셀에 대한 프로토콜 단위 교차검증 예측(log10). exclude_ids 의 셀은 학습에서만 뺀다
    (검증 fold 에는 그대로 포함되므로 같은 셀에서 두 설정을 비교할 수 있다). Batch 1 만 사용한다."""
    _assert_batch1(table)
    X, y, g = select_features(table, feature_set), table[TARGET], table[GROUP_COL]
    drop = set(exclude_ids)
    oof = pd.Series(np.nan, index=table.index)
    for tr, va in group_kfold(n_splits).split(X, y, g):
        keep = [i for i in table.index[tr] if i not in drop]
        est = _fit(make_estimator(name), X.loc[keep], y.loc[keep], g.loc[keep])
        oof.iloc[va] = est.predict(X.iloc[va])
    return oof
