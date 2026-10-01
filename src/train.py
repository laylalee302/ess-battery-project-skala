"""후보 모델 비교 -> 최종 모델. 후보 선정 근거는 DAY1 설계서의 연결표(EDA 발견 -> 결정)를 따른다.

TODO: EDA(Q5 다중공선성, 셀 수) 결과 확정 후 후보/하이퍼파라미터 구체화.
Scaler/feature selection은 반드시 Pipeline 안에 넣어 train fold에서만 fit한다.
"""
from __future__ import annotations

from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED


def candidate_models() -> dict[str, Pipeline]:
    """기준선(선형/정규화) 중심. 트리 계열 등은 EDA 근거가 확인되면 추가."""
    return {
        "ridge": Pipeline([("scale", StandardScaler()), ("model", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))]),
        "elastic_net": Pipeline(
            [("scale", StandardScaler()), ("model", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=SEED))]
        ),
    }
