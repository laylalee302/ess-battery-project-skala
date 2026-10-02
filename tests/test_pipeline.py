"""파이프라인 검증 게이트 (G1~G6). 실행: pytest -q

Batch 1 원본/캐시가 없으면 데이터가 필요한 테스트는 건너뛴다. Batch 2 정답은 어떤 테스트에서도 읽지 않는다.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR
from src.evaluate import mape, pre_eol_cells, regression_report
from src.features import FEATURE_SETS, TARGET, build_feature_table, select_features
from src.preprocess import select_valid
from src.split import assert_no_group_overlap, group_kfold, make_holdout
from src.train import CANDIDATES, GROUP_COL, cross_val_predict_groups, evaluate_candidates, fit_final, make_estimator, select_model

needs_data = pytest.mark.skipif(not (PROCESSED_DIR / "batch1_v2.pkl").exists(), reason="Batch 1 캐시 없음 (data/README.md 참고)")


@pytest.fixture(scope="module")
def batch1():
    from src.load_data import load_batch
    cells, _ = select_valid(load_batch("batch1"))
    return cells, build_feature_table(cells)


# G4 역변환 -------------------------------------------------------------------
def test_g4_mape_on_original_scale():
    y_true = np.array([100.0, 200.0])
    pred_log = np.log10([110.0, 180.0])           # +10%, -10%
    assert mape(y_true, 10 ** pred_log) == pytest.approx(10.0)


def test_gap_sign_rules():
    r = regression_report(10.0, 12.0, 20.0, target=9.1).set_index("구분")["MAPE (%)"]
    assert r["Gap (Train-Valid)"] == pytest.approx(2.0)
    assert r["Gap (Valid-Test)"] == pytest.approx(8.0)
    assert r["Gap (Target-Test)"] == pytest.approx(20.0 - 9.1)


# G1 분할 -----------------------------------------------------------------------
@needs_data
def test_g1_holdout_split(batch1):
    _, table = batch1
    tr, va = make_holdout(table)
    assert len(tr) == 36 and len(va) == 10
    assert table.loc[va, GROUP_COL].nunique() == 5 and table.loc[tr, GROUP_COL].nunique() == 18
    assert_no_group_overlap(table.loc[tr, GROUP_COL], table.loc[va, GROUP_COL])
    tr2, va2 = make_holdout(table)                 # 난수 없음: 항상 같은 분할
    assert list(va) == list(va2)


@needs_data
def test_g1_cv_folds_have_no_shared_protocol(batch1):
    _, table = batch1
    tr, _ = make_holdout(table)
    t = table.loc[tr]
    for a, b in group_kfold(5).split(t, t[TARGET], t[GROUP_COL]):
        assert_no_group_overlap(t.iloc[a][GROUP_COL], t.iloc[b][GROUP_COL])


def test_g1_overlap_is_detected():
    with pytest.raises(AssertionError):
        assert_no_group_overlap(["a", "b"], ["b", "c"])


# G2 Batch 2 격리 ---------------------------------------------------------------
def test_g2_training_rejects_non_batch1_cells():
    t = pd.DataFrame({"batch": ["batch1", "batch2"], "policy": ["p", "q"]}, index=["x", "y"])
    with pytest.raises(AssertionError):
        evaluate_candidates(t, ["x"], ["y"])
    with pytest.raises(AssertionError):
        fit_final("ols", "A", t)


# G3 누수 ----------------------------------------------------------------------
def test_g3_scaler_is_inside_pipeline_and_fit_on_train_only():
    for name in ("ols", "ridge", "elastic_net", "ridge_quad", "huber"):
        est = make_estimator(name)
        pipe = est.estimator if hasattr(est, "estimator") else est
        assert isinstance(pipe, Pipeline) and any(isinstance(s, StandardScaler) for _, s in pipe.steps), name
    rng = np.random.default_rng(0)
    X_tr, X_te = pd.DataFrame(rng.normal(0, 1, (30, 2))), pd.DataFrame(rng.normal(5, 1, (10, 2)))
    y = pd.Series(rng.normal(size=30))
    pipe = CANDIDATES["ols"]["pipe"]().fit(X_tr, y)
    assert np.allclose(pipe.named_steps["scale"].mean_, X_tr.mean().values)   # 평가 데이터 평균이 섞이지 않음
    assert not np.allclose(pipe.named_steps["scale"].mean_, X_te.mean().values)


# G5 피처 -----------------------------------------------------------------------
@needs_data
def test_g5_feature_sets(batch1):
    _, table = batch1
    assert list(select_features(table, "A").columns) == ["dq_log10_var", "dq_kurtosis"]
    assert select_features(table, "B").shape[1] == 8
    assert not table[FEATURE_SETS["B"]].isna().any().any()
    assert len(table) == 46


@needs_data
def test_g5_features_match_eda_report(batch1):
    """DAY 1 보고서의 Batch 1 Spearman: log10(분산) -0.85, 첨도 +0.31 (EDA 노트북과 같은 정의인지 확인)."""
    _, table = batch1
    assert stats.spearmanr(table["dq_log10_var"], table[TARGET])[0] == pytest.approx(-0.85, abs=0.01)
    assert stats.spearmanr(table["dq_kurtosis"], table[TARGET])[0] == pytest.approx(0.31, abs=0.01)


@needs_data
def test_pre_eol_cells_count(batch1):
    cells, _ = batch1
    assert len(pre_eol_cells(cells)) == 9      # 보고서: EOL 도달 전 측정이 끝난 Batch 1의 9셀


# G6 재현성 ---------------------------------------------------------------------
@needs_data
def test_g6_reproducible(batch1):
    _, table = batch1
    tr, _ = make_holdout(table)
    t = table.loc[tr]
    X = select_features(t, "A")
    a, fa = cross_val_predict_groups("ridge_quad", X, t[TARGET], t[GROUP_COL])
    b, fb = cross_val_predict_groups("ridge_quad", X, t[TARGET], t[GROUP_COL])
    assert np.allclose(a, b) and fa == fb


def test_select_model_prefers_simpler_within_one_se():
    comp = pd.DataFrame([
        {"model": "gb", "feature_set": "A", "heldout_mape": 10.0, "heldout_se": 1.0, "complexity": 14, "reference_only": False},
        {"model": "ridge", "feature_set": "A", "heldout_mape": 10.8, "heldout_se": 1.0, "complexity": 4, "reference_only": False},
        {"model": "baseline", "feature_set": "A", "heldout_mape": 5.0, "heldout_se": 1.0, "complexity": 0, "reference_only": True},
    ])
    assert select_model(comp)["model"] == "ridge"   # 1 SE 이내의 단순한 모델, 기준선은 선택 대상이 아님
