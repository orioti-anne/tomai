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

# 1. 환경 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path: sys.path.append(project_root)

from smartfarm import db, create_app

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------
# 2. 성능이 검증된 V3 로직 (Back to Basics)
# --------------------------------------------------
def load_and_preprocess_v3_final():
    print("🔍 [V3-Final] 검증된 데이터 로드 로직 실행...")
    query = text("""
                 SELECT P.PRICE_DATE, P.PRICE_PER_KG, W.AVG_TEMP
                 FROM KAMIS_TOMATO_PRICE P
                          LEFT JOIN WEATHER_INDEX W ON P.PRICE_DATE = W.W_DATE
                 WHERE P.ITEM_NAME = '완숙토마토'
                 ORDER BY P.PRICE_DATE ASC
                 """)

    df = pd.read_sql(query, db.engine)
    df.columns = [c.upper() for c in df.columns]
    df["PRICE_DATE"] = pd.to_datetime(df["PRICE_DATE"])

    # 시계열 연속성 확보
    df = df.set_index("PRICE_DATE").resample('D').asfreq().reset_index()
    df["PRICE_PER_KG"] = df["PRICE_PER_KG"].ffill().bfill()
    df["AVG_TEMP"] = df["AVG_TEMP"].ffill().bfill()

    # 필수 피처 생성
    df["MONTH"] = df["PRICE_DATE"].dt.month
    week = df["PRICE_DATE"].dt.isocalendar().week.astype(int)
    df["WEEK_SIN"] = np.sin(2 * np.pi * week / 52)
    df["WEEK_COS"] = np.cos(2 * np.pi * week / 52)

    # 추세 및 기상 (V3의 핵심)
    df["MA_30D"] = df["PRICE_PER_KG"].shift(1).rolling(30).mean()
    df["MA_90D"] = df["PRICE_PER_KG"].shift(1).rolling(90).mean()

    prev_year = df["PRICE_PER_KG"].shift(365)
    df["YOY_RATIO"] = (df["PRICE_PER_KG"].shift(1) / prev_year).clip(0.5, 1.5).fillna(1.0)

    df["TEMP_MA_30D"] = df["AVG_TEMP"].shift(1).rolling(30).mean()
    df["GDD_30D"] = (df["AVG_TEMP"] - 10).clip(lower=0).shift(1).rolling(30).sum()

    # 타겟: 100일 뒤 시세 (Log 변환 유지)
    df["TARGET"] = np.log1p(df["PRICE_PER_KG"].shift(-100))

    return df.dropna().reset_index(drop=True)


# --------------------------------------------------
# 3. 모델 학습 및 결과 출력/저장
# --------------------------------------------------
def train_and_report_v3_final():
    app = create_app(enable_scheduler=False)

    with app.app_context():
        df = load_and_preprocess_v3_final()

        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        feature_cols = [
            "MA_30D", "MA_90D", "YOY_RATIO", "GDD_30D",
            "TEMP_MA_30D", "WEEK_SIN", "WEEK_COS", "MONTH"
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), feature_cols)
            ]
        )

        # V3에서 가장 높은 성능을 냈던 하이퍼파라미터
        model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )

        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        print(f"🚀 [V3-Final] 학습 시작 (데이터: {len(df)}건)")
        pipeline.fit(train_df[feature_cols], train_df["TARGET"])

        # 평가 및 저장
        pred_log = pipeline.predict(test_df[feature_cols])
        pred = np.expm1(pred_log)
        y_true = np.expm1(test_df["TARGET"])

        r2 = r2_score(y_true, pred)
        mae = mean_absolute_error(y_true, pred)

        # 파일 저장
        model_dir = os.path.join(project_root, "smartfarm", "ml", "models")
        os.makedirs(model_dir, exist_ok=True)

        # 모델 저장 (v3_final 명칭 사용)
        joblib.dump(pipeline, os.path.join(model_dir, "v3_final_tomato_pipeline.joblib"))

        # 시각화 리포트
        plt.figure(figsize=(15, 7))
        visual_dates = test_df["PRICE_DATE"] + pd.Timedelta(days=100)
        plt.plot(visual_dates, y_true, label="실제 가격", color='gray', alpha=0.4)
        plt.plot(visual_dates, pred, label="V3 Final 예측", color='blue', linewidth=2)
        plt.title(f"토마토 수확기 최종 모델 V3 (R2: {r2:.4f}, MAE: {mae:.2f}원)")
        plt.legend();
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(model_dir, "v3_final_report.png"))

        print(f"\n==================================================")
        print(f"✅ V3 최종 모델 학습 및 저장 완료!")
        print(f"📊 R2 Score: {r2:.4f}")
        print(f"💸 평균 오차: {mae:.2f}원")
        print(f"📂 저장 경로: {model_dir}/v3_final_tomato_pipeline.joblib")
        print(f"==================================================")


if __name__ == "__main__":
    train_and_report_v3_final()