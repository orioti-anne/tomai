import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import text
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# --------------------------------------------------
# 1. 경로 및 환경 설정
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app

# 한글 폰트 설정 (Mac: AppleGothic, Windows: Malgun Gothic)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------
# 2. 데이터 로드 및 고도화 피처 생성
# --------------------------------------------------
def load_and_preprocess():
    print("🔍 1. 데이터 통합 로드 (KAMIS 가격 + 기상 인덱스)...")
    query = text("""
                 SELECT P.PRICE_DATE,
                        P.GRADE,
                        P.PRICE_PER_KG,
                        P.UNIT_KG,
                        W.AVG_TEMP,
                        W.SUNSHINE,
                        W.RAIN,
                        W.HUMID
                 FROM KAMIS_TOMATO_PRICE P
                          JOIN WEATHER_INDEX W ON P.PRICE_DATE = W.W_DATE
                 WHERE P.PRICE_PER_KG IS NOT NULL
                 ORDER BY P.PRICE_DATE ASC
                 """)
    df = pd.read_sql(query, db.engine)
    df.columns = [c.upper() for c in df.columns]
    df["PRICE_DATE"] = pd.to_datetime(df["PRICE_DATE"])

    # 그룹 내 시간 순서 정렬
    df = df.sort_values(["GRADE", "UNIT_KG", "PRICE_DATE"]).reset_index(drop=True)

    print("🔍 2. 고도화 피처 생성 및 단위 보정...")
    group_cols = ["GRADE", "UNIT_KG"]
    g = df.groupby(group_cols)

    # 시계열 기반 피처 (누수 방지 위해 shift(1) 적용)
    df["PREV_1D"] = g["PRICE_PER_KG"].shift(1)
    df["MA_3D"] = g["PRICE_PER_KG"].transform(lambda x: x.shift(1).rolling(3).mean())
    df["MA_7D"] = g["PRICE_PER_KG"].transform(lambda x: x.shift(1).rolling(7).mean())
    df["PRICE_DIFF"] = g["PRICE_PER_KG"].shift(1).diff()

    # 기상 시차 및 이동평균 변수
    df["TEMP_LAG7"] = g["AVG_TEMP"].transform(lambda x: x.shift(7))
    df["SUN_MA7"] = g["SUNSHINE"].transform(lambda x: x.rolling(7).mean())

    # 계절성 변수 (Sine/Cosine 변환)
    week = df["PRICE_DATE"].dt.isocalendar().week.astype(int)
    df["WEEK_SIN"] = np.sin(2 * np.pi * week / 52)
    df["WEEK_COS"] = np.cos(2 * np.pi * week / 52)

    # 타겟 설정: 7일 뒤 1kg 단가
    df["TARGET"] = g["PRICE_PER_KG"].shift(-7)

    return df.dropna(subset=["TARGET", "PREV_1D", "TEMP_LAG7", "MA_7D"]).reset_index(drop=True)


# --------------------------------------------------
# 3. 모델 학습 및 리포트 저장
# --------------------------------------------------
def train_and_report():
    app = create_app(enable_scheduler=False)

    with app.app_context():
        df = load_and_preprocess()

        # 날짜 기준 엄격한 시계열 분할 (8:2)
        unique_dates = sorted(df["PRICE_DATE"].unique())
        split_idx = int(len(unique_dates) * 0.8)
        split_date = unique_dates[split_idx]

        train_df = df[df["PRICE_DATE"] < split_date].copy()
        test_df = df[df["PRICE_DATE"] >= split_date].copy()

        # 학습 및 테스트 기간 추출
        train_start, train_end = train_df["PRICE_DATE"].min(), train_df["PRICE_DATE"].max()
        test_start, test_end = test_df["PRICE_DATE"].min(), test_df["PRICE_DATE"].max()

        feature_cols = [
            "GRADE", "PREV_1D", "MA_3D", "MA_7D", "PRICE_DIFF",
            "AVG_TEMP", "SUNSHINE", "TEMP_LAG7", "SUN_MA7", "WEEK_SIN", "WEEK_COS"
        ]

        # 전처리 및 모델 파이프라인 구성
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["GRADE"]),
                ("num", SimpleImputer(strategy="median"), [c for c in feature_cols if c != "GRADE"])
            ]
        )

        model = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )

        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        print(f"🚀 3. 모델 학습 시작 (검증 기준일: {split_date.date()}, 데이터 수: {len(df)})")
        pipeline.fit(train_df[feature_cols], train_df["TARGET"])

        # 성능 지표 산출
        pred = pipeline.predict(test_df[feature_cols])
        y_true = test_df["TARGET"]
        r2 = r2_score(y_true, pred)
        mae = mean_absolute_error(y_true, pred)
        acc_1000 = np.mean(np.abs(pred - y_true) <= 1000) * 100

        # --- [4. 결과 시각화 및 리포트 이미지 저장] ---
        print("📊 4. 모델 검증 결과 리포트 이미지 생성 중...")
        model_dir = os.path.join(project_root, "models")
        os.makedirs(model_dir, exist_ok=True)

        fig, ax = plt.subplots(2, 1, figsize=(12, 15))

        # 리포트 요약 텍스트 (학습/테스트 기간 포함)
        summary_text = (f"[모델 검증 결과 요약]\n\n"
                        f"- 예측 목표: 7일 뒤 1kg 단가\n"
                        f"- 결정계수 (R2): {r2:.4f}\n"
                        f"- 평균 절대 오차 (MAE): {mae:.2f}원\n"
                        f"- ±1,000원 이내 적중률: {acc_1000:.2f}%\n\n"
                        f"- 학습 기간 (Train): {train_start.date()} ~ {train_end.date()}\n"
                        f"- 테스트 기간 (Test): {test_start.date()} ~ {test_end.date()}")
        ax[0].text(0.1, 0.5, summary_text, fontsize=16, va='center', ha='left', linespacing=1.8)
        ax[0].axis('off')

        # 피처 중요도 차트
        model_obj = pipeline.named_steps["model"]
        encoder = pipeline.named_steps["prep"].named_transformers_["cat"]
        all_features = list(encoder.get_feature_names_out(["GRADE"])) + [c for c in feature_cols if c != "GRADE"]

        importances = pd.Series(model_obj.feature_importances_, index=all_features).sort_values(ascending=True)
        importances.tail(10).plot(kind='barh', ax=ax[1], color='skyblue')
        ax[1].set_title("상위 10개 핵심 예측 피처 (Importance)", fontsize=14)

        plt.tight_layout()
        report_path = os.path.join(model_dir, "tomato_model_report.png")
        plt.savefig(report_path)

        # 모델 파이프라인 저장 (.joblib)
        joblib.dump(pipeline, os.path.join(model_dir, "tomato_price_pipeline.joblib"))

        print(f"\n==================================================")
        print(f"✅ 모델 파이프라인 저장 완료: tomato_price_pipeline.joblib")
        print(f"✅ 결과 리포트 이미지 저장 완료: {report_path}")
        print(f"   (테스트 기간: {test_start.date()} ~ {test_end.date()})")
        print(f"==================================================")


if __name__ == "__main__":
    train_and_report()