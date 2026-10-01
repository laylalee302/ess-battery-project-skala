"""전략 피처 구현. 함수/피처 이름은 DAY1 설계서의 피처 ID(FE1~FE5)와 1:1 대응.

  FE1 ΔQ(V) 100-10 통계 (log10 var, min, mean, skew, kurtosis)  <- EDA Q3
  FE2 Qd 사이클 2~100 선형 기울기/절편, 최대-Qd2                  <- EDA Q2
  FE3 IR: cycle2, min, cycle100 - cycle2                          <- EDA Q5
  FE4 Tavg/Tmax 통계, chargetime 평균                             <- EDA Q4/Q5
  FE5 충전 프로토콜(C-rate) 파생값                                <- EDA Q4

TODO: EDA 결과로 확정/수정 후 구현. 전략에 없는 피처는 추가하지 않는다.
"""
from __future__ import annotations

import pandas as pd


def fe1_delta_q_stats(cell: dict) -> dict:
    raise NotImplementedError


def fe2_capacity_fade(cell: dict) -> dict:
    raise NotImplementedError


def fe3_internal_resistance(cell: dict) -> dict:
    raise NotImplementedError


def fe4_temperature_chargetime(cell: dict) -> dict:
    raise NotImplementedError


def fe5_charge_protocol(cell: dict) -> dict:
    raise NotImplementedError


FEATURE_FUNCS = {
    "FE1": fe1_delta_q_stats,
    "FE2": fe2_capacity_fade,
    "FE3": fe3_internal_resistance,
    "FE4": fe4_temperature_chargetime,
    "FE5": fe5_charge_protocol,
}


def build_feature_table(cells: dict, use: tuple[str, ...] = tuple(FEATURE_FUNCS)) -> pd.DataFrame:
    """셀 단위 피처 테이블. index=cell_id, columns=피처, 마지막 열 cycle_life."""
    rows = {}
    for cell_id, cell in cells.items():
        row: dict = {}
        for fid in use:
            row.update(FEATURE_FUNCS[fid](cell))
        row["cycle_life"] = cell["cycle_life"]
        rows[cell_id] = row
    return pd.DataFrame.from_dict(rows, orient="index")
