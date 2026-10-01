"""데이터 분할. 누수 방지 원칙:
  - 분할 단위는 '셀' (행 = 셀 1개이므로 셀이 섞이지 않음)
  - 같은 충전 프로토콜이 train/valid에 갈라지지 않도록 group = 프로토콜
  - Batch 2(Test)는 모델 선택/튜닝에 사용하지 않는다 (최종 평가 1회)
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .config import SEED


def holdout_by_group(X: pd.DataFrame, groups, test_size: float = 0.2, seed: int = SEED):
    """Batch 1 내부 Train/Valid(Hold-out) 분리. 반환: (train_idx, valid_idx)"""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(X, groups=groups))


def group_kfold(n_splits: int = 5) -> GroupKFold:
    """Batch 1 CV용. fit/split 시 groups 전달."""
    return GroupKFold(n_splits=n_splits)
