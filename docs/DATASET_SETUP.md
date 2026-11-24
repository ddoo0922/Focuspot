# 데이터 출처와 활용 범위

| 데이터 | 폴더 | 출처 | 라이선스 | 활용 사유·방법 |
|---|---|---|---|---|
| KAMP 용접기 AI 데이터셋 | `01. KAMP_용접기 AI 데이터셋/2. 용접기 AI 데이터셋/` | 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), 용접기 AI 데이터셋 및 분석실습 가이드북, 2020. www.kamp-ai.kr | 대회 제공 데이터 | 분석 대상. 학습과 예측에 사용 |
| Stellantis N.V. (Politecnico di Torino 공개 데이터) | `03_Stellantis N.V._RSW/data/` | Ciravegna, G., Galante, F., Giordano, D., Cerquitelli, T., Mellia, M. (2024). Fault Prediction in Resistance Spot Welding: A Comparison of Machine Learning Approaches. Electronics 13(18):3693. doi:10.3390/electronics13183693. 데이터 : github.com/smartdatapolito/resistance_spot_welding_dataset | 데이터 CC BY-NC 4.0 (비상업적 이용), 논문 CC BY 4.0 | 타점 라벨이 있는 같은 공정 데이터. 라벨을 숨기고 날짜별 불량 개수만 남겨 KAMP와 같은 조건을 만든 뒤 숨긴 라벨로 성능 채점. KAMP 학습 데이터에 합치지 않음 |
| Querétaro | `02_Queretaro_RSW/data/Queretaro_RSW_v3/` | Domínguez, L. A. 외 (2025). Resistance Spot Welding Insights: A Dataset Integrating Process Parameters, Infrared, and Surface Imaging. Mendeley Data V3. doi:10.17632/rwh8kjzdch.3 | CC BY 4.0 | 가압력·통전시간과 너깃 지름·날림의 관계를 확인하는 물리 방향 근거. 학습·성능 평가에는 쓰지 않음. `Data_RSW.csv`만 포함(이미지 제외) |

- 대회 제출본에는 외부 데이터 원본을 포함했으나, 이 공개 저장소에는 원본 데이터를 포함하지 않는다. 출처 문서만 보존하며 데이터는 제공처에서 별도로 준비한다. Stellantis N.V. 원본 README는 `03_Stellantis N.V._RSW/data/README.md`에 보존했다.
- 모든 외부 데이터는 대회 분석 목적(비상업적)으로만 사용했다.
