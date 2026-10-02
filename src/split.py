"""데이터 분할. 누수 방지 원칙:
  - 분할 단위는 '셀' (행 = 셀 1개)이며, 같은 충전 프로토콜이 train/valid에 갈라지지 않도록 group = 프로토콜(policy)
  - Valid(Hold-out)는 프로토콜 평균 수명으로 층화해 결정적으로 고른다 (난수 없음, 재현 가능)
  - Batch 2(Test)는 모델 선택/튜닝에 사용하지 않는다 (최종 평가 1회)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HOLDOUT_START = 2   # 프로토콜을 평균 수명 순으로 정렬했을 때 세 번째부터
HOLDOUT_STEP = 5    # 5개마다 1개 -> 23개 프로토콜 중 5개(10셀)


def make_holdout(table: pd.DataFrame, group_col: str = "policy", life_col: str = "cycle_life"):
    """Batch 1 내부 Train/Valid(Hold-out) 분리. 반환: (train_ids, valid_ids) = 셀 index 배열.

    프로토콜별 평균 수명 순으로 정렬한 뒤 HOLDOUT_START 번째부터 HOLDOUT_STEP 개마다 하나를 Valid로 고른다.
    짧은 수명부터 긴 수명까지 골고루 포함된다.
    """
    mean_life = table.groupby(group_col)[life_col].mean().sort_values(kind="stable")
    valid_groups = mean_life.index[HOLDOUT_START::HOLDOUT_STEP]
    is_valid = table[group_col].isin(valid_groups).to_numpy()
    return table.index[~is_valid].to_numpy(), table.index[is_valid].to_numpy()


def group_kfold(n_splits: int = 5) -> GroupKFold:
    """Train CV용(그룹 = 프로토콜). split(X, y, groups) 에 groups 를 전달한다."""
    return GroupKFold(n_splits=n_splits)


def assert_no_group_overlap(train_groups, valid_groups) -> None:
    """Train/Valid 에 같은 프로토콜이 없는지 확인한다. 겹치면 예외."""
    overlap = set(np.asarray(train_groups)) & set(np.asarray(valid_groups))
    if overlap:
        raise AssertionError(f"프로토콜이 train/valid 에 모두 있음: {sorted(overlap)}")
