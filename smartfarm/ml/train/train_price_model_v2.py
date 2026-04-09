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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# --------------------------------------------------
# 1. 경로 및 환경 설정
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------
# 2. 데이터 로드 및 피처 엔지니어링 (V3 신규 지표 추가)
# --------------------------------------------------
def load_and_preprocess():
    print("🔍 1. 데이터 로드 및 시계열 연속성 확보...")
    query = text("""
                 SELECT P.PRICE_DATE,
                        P.PRICE_PER_KG,
                        W.AVG_TEMP,
                        W.SUNSHINE,
                        W.RAIN,
                        W.HUMID
                 FROM KAMIS_TOMATO_PRICE P
                          LEFT JOIN WEATHER_INDEX W ON P.PRICE_DATE = W.W_DATE
                 WHERE P.ITEM_NAME = '완숙토마토'
                 ORDER BY P.PRICE_DATE ASC
                 """)

    df = pd.read_sql(query, db.engine)
    df.columns = [c.upper() for c in df.columns]
    df["PRICE_DATE"] = pd.to_datetime(df["PRICE_DATE"])

    # 날짜 빈틈 채우기
    all_dates = pd.date_range(start=df["PRICE_DATE"].min(), end=df["PRICE_DATE"].max(), freq='D')
    df = df.set_index("PRICE_DATE").reindex(all_dates).rename_axis("PRICE_DATE").reset_index()

    # 결측치 보간
    df["PRICE_PER_KG"] = df["PRICE_PER_KG"].interpolate(method='linear').ffill()
    df[['AVG_TEMP', 'SUNSHINE', 'RAIN', 'HUMID']] = df[['AVG_TEMP', 'SUNSHINE', 'RAIN', 'HUMID']].ffill()

    print("🔍 2. V3 신규 고도화 피처 생성 (누적 기상 및 변동성)...")

    # [추가] 가격 변동성 지표: 최근 7일간 가격이 얼마나 요동쳤는가 (Standard Deviation)
    df["PRICE_VOL_7D"] = df["PRICE_PER_KG"].shift(1).rolling(7).std()

    # [추가] 누적 강수량: 최근 7일간 총 강수량 (출하 타격 지표)
    df["RAIN_SUM_7D"] = df["RAIN"].shift(1).rolling(7).sum()

    # [추가] 평균 일조시간: 최근 10일간 평균 일조시간 (생육 타격 지표)
    df["SUN_AVG_10D"] = df["SUNSHINE"].shift(1).rolling(10).mean()

    # 기존 핵심 피처
    df["PREV_1D"] = df["PRICE_PER_KG"].shift(1)
    df["MA_7D"] = df["PRICE_PER_KG"].shift(1).rolling(7).mean()
    df["MA_30D"] = df["PRICE_PER_KG"].shift(1).rolling(30).mean()
    df["TEMP_LAG14"] = df["AVG_TEMP"].shift(14)

    # 계절성
    df["MONTH"] = df["PRICE_DATE"].dt.month
    week = df["PRICE_DATE"].dt.isocalendar().week.astype(int)
    df["WEEK_SIN"] = np.sin(2 * np.pi * week / 52)
    df["WEEK_COS"] = np.cos(2 * np.pi * week / 52)
    df["YEAR"] = df["PRICE_DATE"].dt.year

    # 타겟: 7일 뒤 가격
    df["TARGET"] = df["PRICE_PER_KG"].shift(-7)

    return df.dropna().reset_index(drop=True)


# --------------------------------------------------
# 3. 모델 학습 및 리포트 생성
# --------------------------------------------------
def train_and_report():
    app = create_app(enable_scheduler=False)

    with app.app_context():
        df = load_and_preprocess()

        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # V3 피처 리스트
        feature_cols = [
            "PREV_1D", "MA_7D", "MA_30D", "PRICE_VOL_7D",  # 가격 관련
            "AVG_TEMP", "TEMP_LAG14", "RAIN_SUM_7D", "SUN_AVG_10D",  # 기상 관련
            "WEEK_SIN", "WEEK_COS", "MONTH", "YEAR"  # 계절성 관련
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), feature_cols)
            ]
        )

        model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.02,  # 조금 더 보수적으로 학습 (0.03 -> 0.02)
            max_depth=5,  # 과적합 방지를 위해 깊이 제한 (6 -> 5)
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        print(f"🚀 3. V3 모델 학습 시작 (데이터 수: {len(df)}건)")
        pipeline.fit(train_df[feature_cols], train_df["TARGET"])

        # 성능 평가
        pred = pipeline.predict(test_df[feature_cols])
        y_true = test_df["TARGET"]
        r2 = r2_score(y_true, pred)
        mae = mean_absolute_error(y_true, pred)

        # 결과 저장
        model_dir = os.path.join(project_root, "models")
        os.makedirs(model_dir, exist_ok=True)

        # 그래프 생성
        plt.figure(figsize=(15, 7))
        plt.plot(test_df["PRICE_DATE"], y_true, label="실제 가격", color='steelblue', alpha=0.8)
        plt.plot(test_df["PRICE_DATE"], pred, label="V3 예측 가격", color='darkorange', linestyle='--', alpha=0.9)
        plt.title(f"완숙토마토 7일 뒤 가격 예측 V3 (R2: {r2:.4f}, MAE: {mae:.2f})")
        plt.xlabel("날짜")
        plt.ylabel("가격 (1kg)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        report_path = os.path.join(model_dir, "v3_tomato_report.png")
        plt.savefig(report_path)
        joblib.dump(pipeline, os.path.join(model_dir, "v3_tomato_price_pipeline.joblib"))

        print(f"\n==================================================")
        print(f"✅ V3 모델 갱신 완료! MAE: {mae:.2f}원")
        print(f"✅ 리포트 확인: {report_path}")
        print(f"==================================================")


if __name__ == "__main__":
    train_and_report()