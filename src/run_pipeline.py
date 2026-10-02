"""전체 파이프라인: 데이터 -> 피처 -> 분할 -> 후보 비교/선택(Batch 1만) -> 최종 학습 -> Batch 2 평가 1회 -> results/.

실행: python -m src.run_pipeline

순서가 곧 누수 방지 장치다.
  develop()      Batch 1 만 사용한다. 분할, 후보 비교, 모델 선택까지 끝내고 결과를 저장한다.
  final_test()   선택이 끝난 뒤에야 Batch 2 를 읽고, 최종 모델로 1회 평가한다. 이후 모델을 바꾸지 않는다.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from .config import RESULTS_DIR
from .evaluate import (cell_errors, error_summary, group_errors, mape, out_of_range, positive_dq_cells,
                       pre_eol_cells, regression_report, save_report)
from .features import FEATURE_SETS, TARGET, build_feature_table, select_features
from .load_data import load_batch
from .preprocess import select_valid
from .split import assert_no_group_overlap, make_holdout
from .train import CANDIDATES, GROUP_COL, cv_predict_excluding, evaluate_candidates, fit_final, select_model


def _best_params(est) -> dict:
    return dict(getattr(est, "best_params_", {}))


def save_exclusions() -> pd.DataFrame:
    """제외된 셀과 사유 (Batch 1~3 전체 데이터 점검 결과). 학습/평가에는 Batch 1, 2 의 유효 셀만 사용한다."""
    frames = [select_valid(load_batch(b))[1] for b in ("batch1", "batch2", "batch3")]
    ex = pd.concat(frames, ignore_index=True)
    ex.to_csv(RESULTS_DIR / "excluded_cells.csv", index=False, encoding="utf-8-sig")
    return ex


def develop() -> dict:
    """Batch 1 만으로 분할, 후보 비교, 모델 선택을 수행한다."""
    cells, _ = select_valid(load_batch("batch1"))
    table = build_feature_table(cells)
    assert set(table["batch"]) == {"batch1"}

    train_ids, valid_ids = make_holdout(table)
    assert_no_group_overlap(table.loc[train_ids, GROUP_COL], table.loc[valid_ids, GROUP_COL])

    comparison, preds = evaluate_candidates(table, train_ids, valid_ids)
    comparison.to_csv(RESULTS_DIR / "candidate_comparison.csv", index=False, encoding="utf-8-sig")
    chosen = select_model(comparison)
    return {"cells": cells, "table": table, "train_ids": train_ids, "valid_ids": valid_ids,
            "comparison": comparison, "chosen": chosen, "preds": preds}


def final_test(state: dict) -> dict:
    """선택된 모델을 Batch 1 전체로 학습해 Batch 2 를 1회 평가한다."""
    chosen, table = state["chosen"], state["table"]
    name, fs = chosen["model"], chosen["feature_set"]
    model = fit_final(name, fs, table)

    test_cells, _ = select_valid(load_batch("batch2"))
    test_table = build_feature_table(test_cells)
    pred = model.predict(select_features(test_table, fs))
    err = cell_errors(test_table, pred)
    test_mape = error_summary(err)["mape"]

    # 분석용 플래그 (모델 선택에는 쓰이지 않음)
    train_policies = set(table["policy"])
    err["newstructure"] = test_table["policy"].str.contains("newstructure")
    err["shares_batch1_protocol"] = test_table["policy"].isin(train_policies)
    err["n_features_out_of_range"] = out_of_range(select_features(table, fs), select_features(test_table, fs))
    err["dq_log10_var"] = test_table["dq_log10_var"]
    err["dq_kurtosis"] = test_table["dq_kurtosis"]
    err["dq_positive"] = err.index.isin(positive_dq_cells(test_cells))
    return {"model": model, "test_cells": test_cells, "test_table": test_table, "err": err, "test_mape": test_mape, "pred": pred}


def sensitivity_pre_eol(state: dict) -> tuple[pd.DataFrame, list[str]]:
    """EOL 전에 측정이 끝난 Batch 1 셀(9셀)을 학습에 포함했을 때와 제외했을 때의 Batch 1 교차검증 성능 비교.
    DAY 1 보고서 2-1 계획대로 Test(Batch 2)는 사용하지 않는다. 평가는 두 경우 모두 같은 셀에서 한다:
    9셀을 뺀 나머지 셀에서의 MAPE, 그리고 9셀 자체의 오차(포함해서 학습한 모델 기준)."""
    chosen, table = state["chosen"], state["table"]
    name, fs = chosen["model"], chosen["feature_set"]
    flagged = pre_eol_cells(state["cells"])
    rest = [i for i in table.index if i not in flagged]
    rows = []
    for label, drop in (("9셀 포함", ()), ("9셀 제외", flagged)):
        oof = cv_predict_excluding(name, fs, table, exclude_ids=drop)
        e_rest = error_summary(cell_errors(table.loc[rest], oof.loc[rest]))
        e_flag = error_summary(cell_errors(table.loc[flagged], oof.loc[flagged]))
        rows.append({"학습 데이터": label, "나머지 37셀 MAPE (%)": e_rest["mape"], "나머지 37셀 편향 (%)": e_rest["bias_pct"],
                     "9셀 MAPE (%)": e_flag["mape"], "9셀 편향 (%)": e_flag["bias_pct"]})
    return pd.DataFrame(rows), flagged


def diagnostic_all_candidates(state: dict, test: dict) -> pd.DataFrame:
    """진단용: 선택되지 않은 후보들도 Batch 1 전체로 학습해 Batch 2 에 적용했을 때의 오차.
    오차가 모델 선택 때문인지 배치 차이 때문인지 구분하기 위한 것이며, 모델 선택/변경에는 쓰지 않는다."""
    table, rows = state["table"], []
    for name, spec in CANDIDATES.items():
        for fs in spec["sets"]:
            pred = fit_final(name, fs, table).predict(select_features(test["test_table"], fs))
            s_ = error_summary(cell_errors(test["test_table"], pred))
            rows.append({"model": name, "feature_set": fs, "Test MAPE (%)": s_["mape"], "편향 (%)": s_["bias_pct"],
                         "최소 예측 수명": float(np.min(10 ** pred))})
    return pd.DataFrame(rows)


def shared_protocol_check(state: dict, test: dict) -> pd.DataFrame:
    """진단용: Batch 1 과 Batch 2 에 모두 있는 충전 프로토콜에서, 실제 수명 변화와 모델 예측 변화를 비교한다."""
    table, test_table, model = state["table"], test["test_table"], test["model"]
    fs = state["chosen"]["feature_set"]
    pred1 = pd.Series(10 ** model.predict(select_features(table, fs)), index=table.index)
    pred2 = pd.Series(10 ** model.predict(select_features(test_table, fs)), index=test_table.index)
    rows = []
    for pol in sorted(set(table["policy"]) & set(test_table["policy"])):
        a, b = table["policy"] == pol, test_table["policy"] == pol
        r = {"policy": pol, "Batch 1 셀": int(a.sum()), "Batch 2 셀": int(b.sum()),
             "Batch 1 실제": table.loc[a, "cycle_life"].mean(), "Batch 1 예측": pred1[a].mean(),
             "Batch 2 실제": test_table.loc[b, "cycle_life"].mean(), "Batch 2 예측": pred2[b].mean()}
        r["실제 변화(배)"] = r["Batch 2 실제"] / r["Batch 1 실제"]
        r["예측 변화(배)"] = r["Batch 2 예측"] / r["Batch 1 예측"]
        rows.append(r)
    return pd.DataFrame(rows)


def batch_input_comparison(train_cells: dict, test_cells: dict) -> pd.DataFrame:
    """진단용(수명 정답 미사용): 두 배치의 입력 신호 비교. '전체 중앙값'과 'Batch 1 과 같은 프로토콜 셀끼리 비교'."""
    def one(c):
        s = c["summary"]
        return {"policy": c["policy"], "초기 용량(Ah)": s["QD"].loc[2:6].median(),
                "초기 내부저항(Ohm)": s["IR"].replace(0, np.nan).loc[2:6].median(),
                "Tavg(C)": s["Tavg"].loc[2:100].median(), "Tmax(C)": s["Tmax"].loc[2:100].median(),
                "충전 시간(분)": s["chargetime"].loc[2:100].median()}
    a, b = (pd.DataFrame({k: one(c) for k, c in cs.items()}).T for cs in (train_cells, test_cells))
    cols = [c for c in a.columns if c != "policy"]
    a[cols], b[cols] = a[cols].astype(float), b[cols].astype(float)
    rows = [{"구분": "전체 중앙값", "프로토콜": "-", **{c: f"{a[c].median():.3f} -> {b[c].median():.3f}" for c in cols}}]
    for pol in sorted(set(a["policy"]) & set(b["policy"])):
        rows.append({"구분": "같은 프로토콜 평균", "프로토콜": pol,
                     **{c: f"{a.loc[a.policy == pol, c].mean():.3f} -> {b.loc[b.policy == pol, c].mean():.3f}" for c in cols}})
    return pd.DataFrame(rows)   # 각 칸: Batch 1 값 -> Batch 2 값


def main() -> None:
    warnings.filterwarnings("ignore")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ex = save_exclusions()
    print(f"제외 셀 {len(ex)}개 -> results/excluded_cells.csv")

    state = develop()
    chosen = state["chosen"]
    name, fs = chosen["model"], chosen["feature_set"]
    print(f"선택: {name} (Set {fs}) | held-out MAPE {chosen['heldout_mape']:.2f}% (SE {chosen['heldout_se']:.2f})")

    test = final_test(state)
    err, table = test["err"], state["table"]
    final_params = _best_params(test["model"])

    report = regression_report(
        float(chosen["cv_mape_mean"]), float(chosen["valid_mape"]), test["test_mape"],
        notes={"train": f"{name}(Set {fs}), 프로토콜 단위 5-fold, 학습 {len(state['train_ids'])}셀, fold 표준편차 {chosen['cv_mape_std']:.2f}",
               "valid": f"프로토콜 {table.loc[state['valid_ids'], GROUP_COL].nunique()}개 {len(state['valid_ids'])}셀 (Train {len(state['train_ids'])}셀로 학습한 모델)",
               "test": f"Batch 2 유효 {len(err)}셀 (Batch 1 {len(table)}셀로 재학습한 최종 모델로 평가)"})
    save_report(report)
    print(report.to_string(index=False))

    err.to_csv(RESULTS_DIR / "predictions_batch2.csv", encoding="utf-8-sig")
    groups = pd.concat({
        "newstructure": group_errors(err, err["newstructure"].map({True: "newstructure 9셀", False: "기존 프로토콜"})),
        "Batch 1과 같은 프로토콜": group_errors(err, err["shares_batch1_protocol"].map({True: "겹침", False: "겹치지 않음"})),
        "ΔQ 양수 구간 있음": group_errors(err, err["dq_positive"].map({True: "ΔQ 양수", False: "그 외"})),
        "학습 피처 범위 밖": group_errors(err, (err["n_features_out_of_range"] > 0).map({True: "범위 밖 있음", False: "범위 안"})),
    })
    groups.to_csv(RESULTS_DIR / "error_groups_batch2.csv", encoding="utf-8-sig")
    err[err["dq_positive"]].to_csv(RESULTS_DIR / "dq_positive_cells_batch2.csv", encoding="utf-8-sig")

    sens, flagged = sensitivity_pre_eol(state)
    sens.to_csv(RESULTS_DIR / "sensitivity_pre_eol.csv", index=False, encoding="utf-8-sig")
    print(sens.round(2).to_string(index=False))

    batch_input_comparison(state["cells"], test["test_cells"]).to_csv(
        RESULTS_DIR / "batch_input_comparison.csv", index=False, encoding="utf-8-sig")
    shared_protocol_check(state, test).to_csv(RESULTS_DIR / "shared_protocol_check.csv", index=False, encoding="utf-8-sig")
    diag = diagnostic_all_candidates(state, test)
    diag.to_csv(RESULTS_DIR / "diagnostic_all_candidates_batch2.csv", index=False, encoding="utf-8-sig")

    with open(RESULTS_DIR / "selected_model.json", "w", encoding="utf-8") as f:
        json.dump({"model": name, "feature_set": fs, "features": FEATURE_SETS[fs], "best_params": final_params,
                   "heldout_mape": float(chosen["heldout_mape"]), "heldout_se": float(chosen["heldout_se"]),
                   "pre_eol_cells": flagged, "valid_cells": list(state["valid_ids"])},
                  f, ensure_ascii=False, indent=2, default=float)


if __name__ == "__main__":
    main()
