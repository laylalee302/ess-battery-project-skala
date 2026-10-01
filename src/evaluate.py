"""성능 평가와 Gap 계산 (노션 리포팅 포맷)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, TARGET_REGRESSION_MAPE


def mape(y_true, y_pred) -> float:
    """MAPE(%). 입력은 원 단위(cycle 수). log10 target이면 10**로 역변환 후 전달."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def regression_report(train_cv: float, valid: float, test: float,
                      target: float = TARGET_REGRESSION_MAPE) -> pd.DataFrame:
    """노션 Regression 리포팅 포맷.

    Gap 부호 규칙 (MAPE는 낮을수록 좋음, 모든 Gap은 '뒤 - 앞' 방향이 (+)면 악화):
      Gap (Train-Valid)  = Valid - Train(CV)  (+) : 과적합 의심
      Gap (Valid-Test)   = Test  - Valid      (+) : 배치간 일반화 저하 의심
      Gap (Target-Test)  = Test  - Target     (+) : 원논문보다 나쁨
    """
    rows = [
        ("Train (Batch 1 CV)", train_cv, ""),
        ("Valid (Batch 1 Hold-out)", valid, ""),
        ("Test (Batch 2)", test, ""),
        ("Gap (Train-Valid)", valid - train_cv, "(+) : 과적합 의심"),
        ("Gap (Valid-Test)", test - valid, "(+) : 배치간 일반화 저하 의심"),
        ("Gap (Target-Test)", test - target, f"Target : 원논문 {target}%"),
    ]
    return pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"])


def save_report(df: pd.DataFrame, name: str = "model_performance.csv") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / name, index=False, encoding="utf-8-sig")
