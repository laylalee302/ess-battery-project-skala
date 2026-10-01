# Data

원본 데이터는 용량(약 8GB) 때문에 git에 포함하지 않습니다.

## 다운로드
- Kaggle: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle
- 아래 파일을 `data/raw/` 에 둡니다 (MATLAB v7.3 `.mat`).

| 파일 | Batch | 용도 |
|---|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` (2.8GB) | Batch 1 | 학습 |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` (1.9GB) | Batch 2 | 테스트 |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` (3.0GB) | Batch 3 | 모델링에는 미사용 (DAY1 EDA 비교용, 지침 확인 필요) |
| `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat` (0.1GB) | extra | 사용 안 함 |

## 캐시
`src/load_data.py` 가 필요한 필드만 추출해 `data/processed/` 에 pkl로 저장합니다 (git 제외).

## 출처
Severson et al. (2019), *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy.
