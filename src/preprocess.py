"""전처리: 유효 셀 선별, 이상 스파이크 처리, 충전 정책 파싱.

규칙은 코드로 명시하고, 제외된 셀은 사유와 함께 표(DataFrame)로 반환해 보고서/README에 그대로 옮긴다.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import EARLY_CYCLES

SPIKE_RATIO = 1.2       # 방전용량이 공칭(초기 중앙값)의 1.2배를 넘으면 측정 스파이크로 간주
NOMINAL_WINDOW = 10     # 공칭 용량 = 초기 10사이클 QD 중앙값

_POLICY_RE = re.compile(r"^(\d+(?:\.\d+)?)C\((\d+)%\)-(\d+(?:\.\d+)?)C")


def parse_policy(policy: str) -> dict:
    """'5.2C(58%)-4C' -> {c1: 5.2, switch_pct: 58, c2: 4.0}.
    '-newstructure', '(SLOWCYCLE' 같은 접미사는 무시. 파싱 불가(VarCharge 등)는 NaN."""
    m = _POLICY_RE.match(policy)
    if not m:
        return {"c1": np.nan, "switch_pct": np.nan, "c2": np.nan}
    return {"c1": float(m.group(1)), "switch_pct": float(m.group(2)), "c2": float(m.group(3))}


def exclusion_reason(cell: dict) -> str | None:
    """제외 사유. 유효하면 None."""
    if np.isnan(cell["cycle_life"]):
        return "cycle_life 없음(NaN): 수명 종료 미도달 또는 특수 실험 셀"
    if cell["n_cycles"] < EARLY_CYCLES:
        return f"사이클 수 {cell['n_cycles']} < {EARLY_CYCLES}: 초기 {EARLY_CYCLES}사이클 피처 계산 불가"
    return None


def clean_spikes(cell: dict) -> dict:
    """QD 이상 스파이크(공칭 대비 SPIKE_RATIO 배 초과)를 NaN 처리한 새 셀을 반환. 삭제하지 않고 해당 점만 NaN."""
    qd = cell["summary"]["QD"]
    nominal = qd.iloc[:NOMINAL_WINDOW].median()
    mask = qd > SPIKE_RATIO * nominal
    out = dict(cell)
    out["summary"] = cell["summary"].copy()
    out["summary"].loc[mask, "QD"] = np.nan
    out["n_spikes"] = int(mask.sum())
    return out


def select_valid(cells: dict) -> tuple[dict, pd.DataFrame]:
    """(유효 셀 dict(스파이크 처리 포함), 제외 셀 사유 표)."""
    valid, excluded = {}, []
    for cell_id, cell in cells.items():
        reason = exclusion_reason(cell)
        if reason is None:
            valid[cell_id] = clean_spikes(cell)
        else:
            excluded.append({"cell_id": cell_id, "batch": cell["batch"], "policy": cell["policy"],
                             "n_cycles": cell["n_cycles"], "reason": reason})
    return valid, pd.DataFrame(excluded, columns=["cell_id", "batch", "policy", "n_cycles", "reason"])


def cell_table(cells: dict) -> pd.DataFrame:
    """셀 단위 요약표 (분포/정책 EDA용). index = cell_id."""
    rows = []
    for cell_id, c in cells.items():
        row = {"cell_id": cell_id, "batch": c["batch"], "cycle_life": c["cycle_life"],
               "policy": c["policy"], "n_cycles": c["n_cycles"], "n_spikes": c.get("n_spikes", 0)}
        row.update(parse_policy(c["policy"]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("cell_id")
