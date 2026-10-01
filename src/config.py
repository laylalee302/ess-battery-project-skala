"""프로젝트 공통 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SEED = 42

BATCH_FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

# 입력으로 사용하는 초기 사이클 구간 (Regression: 초기 100 사이클)
EARLY_CYCLES = 100
DQ_CYCLE_HI = 100  # ΔQ(V) = Q(cycle 100) - Q(cycle 10)
DQ_CYCLE_LO = 10

# 원논문 목표 성능 (노션 과제 기준)
TARGET_REGRESSION_MAPE = 9.1       # %
TARGET_CLASSIFICATION_ACC = 95.1   # %

# Classification 선택 시 장/단수명 기준 (cycle_life >= 550 -> 1)
LONG_LIFE_THRESHOLD = 550
