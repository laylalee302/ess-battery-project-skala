"""전략 피처 구현. 이름과 정의는 DAY 1 보고서(6장 Feature Engineering)와 1:1로 대응하며,
EDA 노트북(01_EDA.ipynb)의 계산 방식과 같다.

  FE1 ΔQ(V) = Qdlin(cycle 100) - Qdlin(cycle 10) 요약: log10(분산), 첨도, 왜도   <- EDA Q3
  FE2 방전 용량: 초기 용량(사이클 2~6 중앙값), 사이클 2~100 선형 기울기          <- EDA Q2
  FE4 온도: 사이클 2~100 Tmax 중앙값                                            <- EDA Q4
  FE5 충전 프로토콜: C2, 전환 지점                                              <- EDA Q4

내부저항(FE3), chargetime, 평균 C-rate, C1, Tavg, ΔQ 평균/최솟값은 보고서에서 제외하기로 했으므로 만들지 않는다.

피처 세트
  Set A (최종 후보): dq_log10_var, dq_kurtosis
  Set B (비교용)   : Set A + dq_skew, qd_init, qd_slope, tmax, c2, switch_pct
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import DQ_CYCLE_HI, DQ_CYCLE_LO, EARLY_CYCLES
from .preprocess import parse_policy

FEATURE_SETS: dict[str, list[str]] = {
    "A": ["dq_log10_var", "dq_kurtosis"],
    "B": ["dq_log10_var", "dq_kurtosis", "dq_skew", "qd_init", "qd_slope", "tmax", "c2", "switch_pct"],
}
TARGET = "log10_cycle_life"


def fe1_delta_q(cell: dict) -> dict:
    """ΔQ(V) = Q_discharge(V)|cycle 100 - Q_discharge(V)|cycle 10 (Qdlin_early 의 인덱스는 사이클-1)."""
    dq = cell["Qdlin_early"][DQ_CYCLE_HI - 1] - cell["Qdlin_early"][DQ_CYCLE_LO - 1]
    return {
        "dq_log10_var": float(np.log10(np.nanvar(dq))),
        "dq_kurtosis": float(stats.kurtosis(dq, nan_policy="omit")),
        "dq_skew": float(stats.skew(dq, nan_policy="omit")),
    }


def fe2_capacity(cell: dict) -> dict:
    """qd_init: 사이클 2~6 방전용량 중앙값. qd_slope: 사이클 2~100 선형 기울기(Ah/100사이클, 스파이크 NaN 제외)."""
    qd = cell["summary"]["QD"]
    x = np.arange(2, EARLY_CYCLES + 1)
    y = qd.loc[2:EARLY_CYCLES].values
    ok = ~np.isnan(y)
    return {
        "qd_init": float(qd.loc[2:6].median()),
        "qd_slope": float(np.polyfit(x[ok], y[ok], 1)[0] * 100),
    }


def fe4_temperature(cell: dict) -> dict:
    return {"tmax": float(cell["summary"]["Tmax"].loc[2:EARLY_CYCLES].median())}


def fe5_protocol(cell: dict) -> dict:
    p = parse_policy(cell["policy"])
    return {"c2": p["c2"], "switch_pct": p["switch_pct"]}


FEATURE_FUNCS = {
    "FE1": fe1_delta_q,
    "FE2": fe2_capacity,
    "FE4": fe4_temperature,
    "FE5": fe5_protocol,
}


def build_feature_table(cells: dict) -> pd.DataFrame:
    """셀 단위 피처 표. index=cell_id. 열: Set B 피처 8개 + log10_cycle_life + cycle_life + policy + batch.

    Set A 는 Set B 의 부분집합이므로 select_features 로 고른다. 피처 열에 결측이 있으면 예외를 낸다.
    """
    rows = {}
    for cell_id, cell in cells.items():
        row: dict = {}
        for fn in FEATURE_FUNCS.values():
            row.update(fn(cell))
        row["cycle_life"] = cell["cycle_life"]
        row[TARGET] = float(np.log10(cell["cycle_life"]))
        row["policy"] = cell["policy"]
        row["batch"] = cell["batch"]
        rows[cell_id] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    cols = FEATURE_SETS["B"]
    if table[cols].isna().any().any():
        bad = table[cols].isna().sum()
        raise ValueError(f"피처 결측: {bad[bad > 0].to_dict()}")
    return table


def select_features(table: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    return table[FEATURE_SETS[feature_set]]
