# tomAI — where tomato meets AI

스마트팜 토마토 재배 데이터를 기반으로 생산량·시세를 예측하고 최적 출하 시점을 추천하는 AI 솔루션

**서비스 주소: [tomai.orioti.com](https://tomai.orioti.com)**

## 주요 기능

- **생산량 및 시세 예측** — 재배·생육·환경 데이터 기반 예상 수확량·출하단가·매출 예측
- **생육분석** — 개체별 생육 데이터 관리, 환경 기반 AI 생육 예측 및 최적 환경 조건 추천
- **환경제어** — 가상 센서 기반 실시간 환경 모니터링 및 이상 알림
- **AI Vision** — YOLOv8 기반 품질 분류·병해 탐지·출하 선별 이미지/영상 분석
- **대시보드** — 재배별 예측 결과, 수확 타임라인, AI 신뢰도, 지역 날씨 통합 제공

## 기술 스택

- Backend: Python 3.11, Flask
- Database: PostgreSQL
- ML: Scikit-learn, XGBoost, LightGBM
- Vision: YOLOv8 (Ultralytics)
- External API: 기상청 Open API, KAMIS Open API

## 모델 파일

용량 문제로 학습된 모델 가중치(`*.joblib`, `*.pt`, `*.mlpackage`)는 저장소에 포함하지 않습니다. 아래 스크립트로 재생성하세요.

- 시세 예측: `smartfarm/ml/train/train_price_model_v7.py`
- 수확량 예측: `smartfarm/ml/train/train_yield_model_v5.py`
- 환경 추천: `smartfarm/ml/train/train_env_recommendation_model.py`
- Vision(YOLOv8): 별도의 라벨링된 이미지 데이터셋으로 학습 후 `smartfarm/dl/models/`에 배치. Apple Silicon 가속(CoreML) 변환은 `ultralytics`의 `YOLO(...).export(format='coreml')`를 이용하세요.

## 관련 저장소

- [tomai-chat](https://github.com/orioti-anne/tomai-chat) — LLM 토마토 재배 상담 서버
- [slm-pipeline](https://github.com/orioti-anne/slm-pipeline) — SLM 멀티태스크 NLP 서버
