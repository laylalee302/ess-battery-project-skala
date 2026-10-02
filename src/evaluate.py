"""성능 평가와 Gap 계산 (노션 리포팅 포맷)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DQ_CYCLE_HI, DQ_CYCLE_LO, RESULTS_DIR, TARGET_REGRESSION_MAPE


def mape(y_true, y_pred) -> float:
    """MAPE(%). 입력은 원 단위(cycle 수). log10 target이면 10**로 역변환 후 전달."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def regression_report(train_cv: float, valid: float, test: float,
                      target: float = TARGET_REGRESSION_MAPE, notes: dict | None = None) -> pd.DataFrame:
    """노션 Regression 리포팅 포맷.

    Gap 부호 규칙 (MAPE는 낮을수록 좋음, 모든 Gap은 '뒤 - 앞' 방향이 (+)면 악화):
      Gap (Train-Valid)  = Valid - Train(CV)  (+) : 과적합 의심
      Gap (Valid-Test)   = Test  - Valid      (+) : 배치간 일반화 저하 의심
      Gap (Target-Test)  = Test  - Target     (+) : 원논문보다 나쁨
    """
    notes = notes or {}
    rows = [
        ("Train (Batch 1 CV)", train_cv, notes.get("train", "")),
        ("Valid (Batch 1 Hold-out)", valid, notes.get("valid", "")),
        ("Test (Batch 2)", test, notes.get("test", "")),
        ("Gap (Train-Valid)", valid - train_cv, "(+) : 과적합 의심"),
        ("Gap (Valid-Test)", test - valid, "(+) : 배치간 일반화 저하 의심"),
        ("Gap (Target-Test)", test - target, f"Target : 원논문 {target}%"),
    ]
    df = pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"])
    df["MAPE (%)"] = df["MAPE (%)"].round(2)
    return df


def save_report(df: pd.DataFrame, name: str = "model_performance.csv") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / name, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 오류 분석 (DAY 2)
# ---------------------------------------------------------------------------
PRE_EOL_SOH = 85.0   # 마지막 10사이클 방전용량 중앙값/공칭이 85%를 넘으면 EOL(80%) 전에 측정이 끝난 셀
PRE_EOL_WINDOW = 10  # DAY 1 보고서와 같은 정의


def pre_eol_cells(cells: dict) -> list[str]:
    """측정이 EOL 도달 전에 끝나 cycle_life 가 실제보다 짧게 기록되었을 가능성이 높은 셀 id (Batch 1 용).

    SOH_end = (cycle_life 직전 10사이클 QD 중앙값) / (초기 10사이클 QD 중앙값) * 100 > PRE_EOL_SOH.
    """
    out = []
    for cell_id, c in cells.items():
        qd = c["summary"]["QD"]
        nominal = qd.iloc[:10].median()
        last = min(int(c["cycle_life"]), len(qd))
        if qd.iloc[max(last - PRE_EOL_WINDOW, 0):last].median() / nominal * 100 > PRE_EOL_SOH:
            out.append(cell_id)
    return out


def cell_errors(table: pd.DataFrame, pred_log10) -> pd.DataFrame:
    """셀별 예측 결과. error_pct 는 (예측-실제)/실제*100 이며 (+)는 수명을 길게 예측했다는 뜻."""
    pred = pd.Series(np.asarray(pred_log10, dtype=float), index=table.index)
    true = table["cycle_life"]
    pred_life = 10 ** pred
    out = pd.DataFrame({
        "policy": table["policy"], "cycle_life": true, "pred_cycle_life": pred_life,
        "error_cycles": pred_life - true, "error_pct": (pred_life - true) / true * 100,
    })
    out["ape"] = out["error_pct"].abs()
    return out


def error_summary(err: pd.DataFrame) -> dict:
    """MAPE, 편향(부호 있는 평균 오차), 산포(오차의 표준편차), 평균 절대 오차(사이클)."""
    return {"n": len(err), "mape": float(err["ape"].mean()), "bias_pct": float(err["error_pct"].mean()),
            "spread_pct": float(err["error_pct"].std(ddof=1)) if len(err) > 1 else float("nan"),
            "mae_cycles": float(err["error_cycles"].abs().mean())}


def group_errors(err: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """labels 값(예: 'newstructure', 'Batch 1과 같은 프로토콜')별 error_summary 표."""
    return pd.DataFrame({k: error_summary(err.loc[idx]) for k, idx in err.groupby(labels).groups.items()}).T


def out_of_range(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.Series:
    """학습 피처의 [최소, 최대] 밖에 있는 피처 개수 (셀별). 학습 데이터에 없던 구간 여부를 본다."""
    lo, hi = X_train.min(), X_train.max()
    return ((X_test < lo) | (X_test > hi)).sum(axis=1)


def positive_dq_cells(cells: dict, threshold: float = 0.002) -> list[str]:
    """ΔQ(V) = Q(cycle 100) - Q(cycle 10) 가 양수(> threshold Ah)가 되는 구간이 있는 셀 id (EDA 3장과 같은 정의)."""
    out = []
    for cell_id, c in cells.items():
        dq = c["Qdlin_early"][DQ_CYCLE_HI - 1] - c["Qdlin_early"][DQ_CYCLE_LO - 1]
        if np.nanmax(dq) > threshold:
            out.append(cell_id)
    return out
