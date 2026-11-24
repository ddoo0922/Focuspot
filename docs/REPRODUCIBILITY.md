> 이 문서는 대회 제출 당시 실행·실험 안내입니다. 아래 경로는 저장소 루트를 기준으로 합니다. 공개용 저장소에는 원본 데이터와 pickle 캐시가 포함되지 않습니다. 데이터 준비와 캐시 생성 순서는 [데이터 준비 안내](DATASET_SETUP.md)를 확인하세요.

# 약지도학습 기반 점용접 불량 예측 및 검사예산 최적화 플랫폼

제6회 K-인공지능 제조데이터 분석 경진대회 · 팀 CACADACA (신현호, 김두용)

날짜별 불량 개수만 있는 KAMP 용접기 데이터로 타점별 불량 위험 순위를 만들고, 하루 검사예산(기본 30%)만큼 위험 순위 상위 타점부터 검사한다.
최종 모델은 raw PVC(날짜별 불량 개수로 학습하는 약지도학습 MLP, 10시드 평균)와 D4(그날 중앙값 대비 이탈 크기 규칙, 학습 없음)의 날짜별 순위 평균이다.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `README.md` | 이 문서 |
| `requirements.txt` | 실행 환경 |
| `테스트데이터_예측결과.csv` | KAMP 11,939타점 예측결과 (아래 설명) |
| `학습용 데이터/` | 학습용 데이터 3종과 출처 (`데이터_출처.md`) : KAMP(`01. KAMP_용접기 AI 데이터셋/`), Querétaro(`02_Queretaro_RSW/`), Stellantis N.V.(`03_Stellantis N.V._RSW/`) |
| `소스코드/run_all.py` | 전체 실행 스크립트 (아래 실행 방법) |
| `소스코드/exp/` | 1차 실험 01~11 (`.py` 원본 + 실행 결과가 담긴 `.ipynb`) : 후보 모델 비교, 입력 비교, KAMP 일자 단위 검증 |
| `소스코드/exp/wl/` | 공용 모듈 ① : 데이터 로딩·타점 요약(data.py), KAMP 전처리·복사 블록 표시(kamp.py), 날짜 묶음 구성(bags.py), 개수 손실 모델(models.py), 예산 지표·동점 처리(metrics.py), 교차검증 실행(runner.py) |
| `소스코드/budget_repro/` | 2차 실험 01~34 (`.py` 원본, 01~31(30_loc_deviation 제외)은 실행 결과가 담긴 `.ipynb` 포함) : 검사예산 평가, raw PVC·D4·결합, 타당성 검증, 물리 근거, 오류분석, 검사예산 30% 재계산·안정 공정범위, 군집화 |
| `소스코드/budget_repro/br/` | 공용 모듈 ② : 단일 모델 사양(spec.py), 검사예산 평가(budget.py), 약지도 모델(weak.py), 타당성 검증(validity.py), 통계(stat.py), 그림 설정(plot.py), 파형 특징(features.py), 타점 라벨 비교 모델(models.py) |
| `소스코드/budget_repro/results/` | 결과표(CSV)와 저장된 점수(pkl) |
| `소스코드/budget_repro/figures/` | 보고서 그림 원본 |
| `소스코드/budget_repro/run_nb.py` | 노트북 1개 실행기 (.py → .ipynb 변환 후 실행) |
| `소스코드/budget_repro/실험기록.md` | 노트북별 목적·결과·판정 |
| `소스코드/EDA/` | 1장 데이터 진단 : KAMP(k1~k2), Stellantis N.V.(p0~p7), Querétaro(q1~q3), 진단 그림(`figures/`)·결과(`results/`) |
| `소스코드/ui/` | 검사예산 최적화 플랫폼 화면(시연용 HTML)과 생성 코드 |

## 실행 방법

```
pip install -r requirements.txt
cd 소스코드/budget_repro
python run_nb.py 06_same_model          # 노트북 1개 실행
cd ..
python run_all.py                        # 2차 실험 01~34 전체 (1차 실험까지: --exp)
python run_all.py 17                     # 17번부터 실행
```

- `exp/01~11`과 `budget_repro/01~29`, `30_combined_components`, `31`은 실행 결과(출력·그림)가 담긴 `.ipynb`를 같은 이름의 `.py`와 함께 넣었다. `exp`와 `budget_repro/01~28`은 실험 PC(GPU)에서 실행한 출력이고, 29·30·31은 저장된 점수로 다시 실행한 출력이다(학습 없음). `06_same_model`의 마지막 그림은 축 범위만 고친 코드로 저장된 결과에서 다시 그렸다. `.ipynb`는 Jupyter Notebook에서 바로 열어 실행할 수 있고, `run_nb.py`·`run_all.py`는 `.py`에서 `.ipynb`를 다시 만들어 실행한다.
- 1장 데이터 진단은 `소스코드/EDA`에서 `python k1_kamp.py`처럼 k1→k2, p0→p7, q1→q3 순서로 실행한다.
- 데이터 경로는 각 코드 파일 위치를 기준으로 한 상대경로(`학습용 데이터/…`)이므로, 압축을 푼 폴더 구조를 그대로 두면 어느 위치에서도 실행된다.
- `run_nb.py`는 `.py`(jupytext 형식)를 `.ipynb`로 바꿔 처음부터 끝까지 실행하고, 출력과 그림을 저장한다.
- GPU가 있으면 자동으로 사용하고 없으면 CPU로 실행한다. 실험은 RTX 4090 2대에서 수행했다.
- raw PVC 두 데이터 학습(5개 모델 × 5시드)이 약 6분, 시드 5개 추가가 약 4분이다(GPU 기준). D4와 29번 이후 분석은 저장된 점수를 써서 수 초~수 분 안에 끝난다.
- `results/`에 저장된 점수(pkl)가 있으므로 학습 없이 17번 이후 노트북만 다시 실행해도 보고서 수치가 나온다.

## 단계별 실행 파일

| 단계 | 실행 파일 | 하는 일 | 산출물 |
|---|---|---|---|
| 전처리 | `exp/wl/kamp.py`, `exp/wl/data.py`, `br/spec.py` (노트북에서 호출) | 원본 xlsx 로딩, 복사 블록 표시, 일별 라벨 정리, 파형 → 타점 요약값, 파생 변수 | `exp/cache/` |
| 학습 | `06_same_model`, `09_final_pipeline`, `11_seed_stability` | raw PVC 10시드 학습 (두 데이터에 같은 사양) | `results/same_model_scores.pkl` |
| 추론 | `17_derived5_rule`, `18_kamp_deep` | D4 계산, raw PVC + D4 점수와 등급 | `results/final_kamp_recommendations_combined.csv` |
| 평가 | `19`, `20`, `23`~`28`, `29`, `31` | 시간순 평가, 타당성 검증, 물리 근거, 오류분석, 검사예산 30% 재계산·안정 공정범위 | `results/*.csv`, `*.md` |
| 결과 생성 | `21_kamp_report_fig`, `29_error_analysis`, `31_budget30_report` | 보고서 그림, 날짜별 검사 계획표 | `figures/*.png`, `results/kamp_daily_inspection_plan.csv` |
| 추가 분석 | `32_clustering`, `33_defect_type_clustering`, `34_tsne_kamp` | 비지도 군집화, 불량 유형 3종 분리 검증, t-SNE | `results/cluster_*.csv`, `results/type_cluster_*.csv` |

## 테스트데이터 예측결과

`테스트데이터_예측결과.csv` (원본 : `소스코드/budget_repro/results/final_kamp_recommendations_combined.csv`에 검사예산 30% 열 추가)

| 열 | 내용 |
|---|---|
| `idx`, `date` | KAMP 원본 행 번호와 작업일 |
| `상태` | `순위 부여` 또는 `품질확인`(복사 블록 1,164행, 위험 순위에서 제외) |
| `raw위험점수_순위백분위`, `D4이탈크기_순위백분위` | 두 모델의 날짜 안 순위 백분위 |
| `결합점수`, `일내순위(결합)` | 최종 점수(두 백분위의 평균)와 그날 안 순위 |
| `추천_5%` ~ `추천_30%` | 검사예산별 우선검사 대상 여부 (30% = 운영 기준) |
| `결합 등급` | A = 그날 불량 개수 c만큼 상위, B = 상위 5c까지, C = 나머지 |
| `의사라벨 등급` | raw PVC 단독 순위 기준 등급 (참고) |

- 라벨이 없는 3/27도 예측을 포함한다(8일로 학습, 등급은 예측 개수 6개 기준).
- 날짜별 요약은 `results/kamp_daily_inspection_plan.csv`, `results/b30_kamp_daily_plan.csv`.

## 재현 확인

- 캐시를 쓰지 않고 새로 학습해 핵심 수치 2개를 다시 계산했다(`22_repro_check`). Stellantis N.V. 시간순 홀드아웃 검사예산 10% 결합 포착 51건 중 17건, KAMP 시드 묶음 간 상위 10% 추천 겹침 0.849로 같았다.
- `requirements.txt`의 버전 조합으로 캐시 없이 원본 데이터부터 다시 계산했을 때(CPU), Stellantis N.V. 수치는 51건 중 17건으로 같았다. KAMP는 CPU 실행에서 점수가 소폭 달라져 시드 묶음 간 상위 10% 겹침이 0.825였다(GPU 0.849, 시드 0 점수의 순위상관 0.99). 보고서 수치는 GPU 결과다.
- 모든 학습에 시드(0~9)를 고정했다. 분할 단위는 날짜로 고정해 같은 날이 학습과 평가에 함께 들어가지 않는다. 점수가 같은 타점은 라벨을 쓰지 않고 기대값으로 처리한다.
