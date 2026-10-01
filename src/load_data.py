"""MATLAB v7.3(.mat) 로딩 -> 셀 단위 표준 구조 + 캐시.

구조/검증 결과는 docs/STRUCTURE_DESIGN.md 의 "데이터 확인 결과" 참고. 요약:
  - mat73.loadmat() 최상위 키: 'batch', 'batch_date'. batch 는 dict-of-lists (셀별 항목)
  - 셀 키: Vdlin, barcode, channel_id, cycle_life, cycles, policy, policy_readable, summary
  - 'MATLAB type not supported: string' 로그 = barcode/channel_id (사용 안 함) -> 로그만 끈다
  - Batch 1 만 summary/cycles 의 index 0 이 빈 행(QD==0, cycles None) -> 제거해서 3개 배치를 정렬
  - cycle_life 는 데이터셋이 준 값을 그대로 사용 (NaN 가능: Batch 2 에 8셀, Batch 3 에 2셀)

표준 셀 구조 (extract_cells 반환값의 값):
  {
    "cell_id": "batch1_c000", "batch": "batch1", "cell_idx": 0,
    "cycle_life": float (NaN 가능), "policy": "5.2C(58%)-4C",
    "n_cycles": int,                     # 정렬 후 실제 사이클 수
    "dropped_placeholder": bool,
    "summary": DataFrame[QD, QC, IR, Tmax, Tavg, Tmin, chargetime], index = 사이클 번호(1부터),
    "Qdlin_early": ndarray (EARLY_CYCLES, 1000), # 사이클 1..EARLY_CYCLES, 없으면 NaN 행
    "Vdlin": ndarray (1000,),
    "profile10": {"I","V","T","t"} 10번째 사이클 원시 시계열 또는 None,
  }
"""
from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd

from .config import BATCH_FILES, DQ_CYCLE_LO, EARLY_CYCLES, PROCESSED_DIR, RAW_DIR

CACHE_VERSION = 2
SUMMARY_FIELDS = {
    "QD": "QDischarge", "QC": "QCharge", "IR": "IR",
    "Tmax": "Tmax", "Tavg": "Tavg", "Tmin": "Tmin", "chargetime": "chargetime",
}


def load_raw_batch(batch: str) -> dict:
    """원본 .mat 를 mat73 으로 로드 (약 15~25초, 메모리 수 GB). 직접 호출보다 load_batch() 사용."""
    import mat73  # 지연 import

    path = RAW_DIR / BATCH_FILES[batch]
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음. data/README.md 참고하여 다운로드하세요.")
    logging.disable(logging.CRITICAL)  # mat73 의 string 필드 미지원 ERROR 로그 억제
    try:
        return mat73.loadmat(str(path))
    finally:
        logging.disable(logging.NOTSET)


def _to_float(x) -> float:
    a = np.asarray(x, dtype=float)
    return float(a.reshape(-1)[0]) if a.size else float("nan")


def extract_cells(raw: dict, batch: str, n_early: int = EARLY_CYCLES) -> dict:
    """raw dict -> {cell_id: 표준 셀 구조}."""
    b = raw["batch"]
    n_cells = len(b["cycle_life"])
    cells: dict = {}
    for i in range(n_cells):
        s = b["summary"][i]
        c = b["cycles"][i]
        arrays = {k: np.asarray(s[v], dtype=float).reshape(-1) for k, v in SUMMARY_FIELDS.items()}

        # Batch 1: index 0 이 빈 행 (QD == 0, cycles None) -> 제거해 사이클 번호를 정렬
        placeholder = bool(arrays["QD"].size and arrays["QD"][0] == 0)
        offset = 1 if placeholder else 0
        summary = pd.DataFrame({k: v[offset:] for k, v in arrays.items()})
        summary.index = np.arange(1, len(summary) + 1)
        summary.index.name = "cycle"

        qdlin_all = c["Qdlin"]
        qdlin = np.full((n_early, 1000), np.nan)
        for k in range(n_early):
            j = k + offset
            if j < len(qdlin_all) and qdlin_all[j] is not None and np.size(qdlin_all[j]) == 1000:
                qdlin[k] = np.asarray(qdlin_all[j], dtype=float).reshape(-1)

        # 충전 전류 패턴 확인용: 10번째 사이클의 원시 I/V/T/t (EDA Q4). 없으면 None
        j10 = DQ_CYCLE_LO - 1 + offset
        profile10 = None
        if j10 < len(c["I"]) and c["I"][j10] is not None:
            profile10 = {k: np.asarray(c[k][j10], dtype=float).reshape(-1) for k in ("I", "V", "T", "t")}

        cell_id = f"{batch}_c{i:03d}"
        cells[cell_id] = {
            "cell_id": cell_id,
            "batch": batch,
            "cell_idx": i,
            "cycle_life": _to_float(b["cycle_life"][i]),
            "policy": str(b["policy_readable"][i]),
            "n_cycles": len(summary),
            "dropped_placeholder": placeholder,
            "summary": summary,
            "Qdlin_early": qdlin,
            "Vdlin": np.asarray(b["Vdlin"][i], dtype=float).reshape(-1),
            "profile10": profile10,
        }
    return cells


def load_batch(batch: str, use_cache: bool = True) -> dict:
    """셀 단위 표준 구조 반환. 첫 호출 시 data/processed/{batch}.pkl 로 캐시 (git 제외)."""
    cache = PROCESSED_DIR / f"{batch}_v{CACHE_VERSION}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    cells = extract_cells(load_raw_batch(batch), batch)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(cells, f)
    return cells


def load_all(batches: tuple[str, ...] = ("batch1", "batch2", "batch3")) -> dict:
    """{batch: {cell_id: 셀}}. 한 배치씩 로드해 메모리 사용을 줄인다."""
    return {b: load_batch(b) for b in batches}
