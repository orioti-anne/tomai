"""
Price Model V5
- 추가: TARGET_DOY_SIN/COS (연중 일자 기반) → 10일 간격 구분
- 추가: PRICE_MA_7D (초단기 시세 흐름)
- 추가: PRICE_TREND_30D (최근 30일 시세 방향성)
- 유지: v4 모든 피처 (MA_30D, MA_90D, YOY_RATIO, GDD_30D, TEMP_MA_30D,
         PRICE_LAG_90D, PRICE_LAG_180D, CURRENT_MONTH_SIN/COS, TARGET_MONTH_SIN/COS)
- 저장: v5_tomato_price_pipeline.joblib
"""
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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def load_and_preprocess_v5():
    print("[V5] 데이터 로드...")
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

    df = df.set_index("PRICE_DATE").resample('D').asfreq().reset_index()
    df["PRICE_PER_KG"] = df["PRICE_PER_KG"].ffill().bfill()
    df["AVG_TEMP"] = df["AVG_TEMP"].ffill().bfill()

    # ── 기존 v4 피처 ─────────────────────────────────────────
    df["MA_30D"] = df["PRICE_PER_KG"].shift(1).rolling(30).mean()
    df["MA_90D"] = df["PRICE_PER_KG"].shift(1).rolling(90).mean()

    prev_year = df["PRICE_PER_KG"].shift(365)
    df["YOY_RATIO"] = (df["PRICE_PER_KG"].shift(1) / prev_year).clip(0.5, 1.5).fillna(1.0)

    df["TEMP_MA_30D"] = df["AVG_TEMP"].shift(1).rolling(30).mean()
    df["GDD_30D"] = (df["AVG_TEMP"] - 10).clip(lower=0).shift(1).rolling(30).sum()

    df["PRICE_LAG_90D"] = df["PRICE_PER_KG"].shift(90)
    df["PRICE_LAG_180D"] = df["PRICE_PER_KG"].shift(180)

    curr_month = df["PRICE_DATE"].dt.month
    df["CURRENT_MONTH_SIN"] = np.sin(2 * np.pi * curr_month / 12)
    df["CURRENT_MONTH_COS"] = np.cos(2 * np.pi * curr_month / 12)

    target_date = df["PRICE_DATE"] + pd.Timedelta(days=100)
    target_month = target_date.dt.month
    df["TARGET_MONTH_SIN"] = np.sin(2 * np.pi * target_month / 12)
    df["TARGET_MONTH_COS"] = np.cos(2 * np.pi * target_month / 12)

    # ── 신규 v5 피처 ─────────────────────────────────────────
    # 연중 일자 기반 (10일 간격 구분 핵심)
    target_doy = target_date.dt.day_of_year
    df["TARGET_DOY_SIN"] = np.sin(2 * np.pi * target_doy / 365)
    df["TARGET_DOY_COS"] = np.cos(2 * np.pi * target_doy / 365)

    # 초단기 시세 흐름 (7일 이동평균)
    df["PRICE_MA_7D"] = df["PRICE_PER_KG"].shift(1).rolling(7).mean()

    # 최근 시세 방향성: (30일 평균 - 90일 평균) / 90일 평균
    df["PRICE_TREND_30D"] = (df["MA_30D"] - df["MA_90D"]) / df["MA_90D"].replace(0, np.nan)

    # 타겟: 100일 뒤 시세 (log 변환)
    df["TARGET"] = np.log1p(df["PRICE_PER_KG"].shift(-100))

    return df.dropna().reset_index(drop=True)


FEATURE_COLS = [
    # v4 피처
    "MA_30D", "MA_90D", "YOY_RATIO", "GDD_30D", "TEMP_MA_30D",
    "PRICE_LAG_90D", "PRICE_LAG_180D",
    "CURRENT_MONTH_SIN", "CURRENT_MONTH_COS",
    "TARGET_MONTH_SIN", "TARGET_MONTH_COS",
    # v5 신규
    "TARGET_DOY_SIN", "TARGET_DOY_COS",
    "PRICE_MA_7D",
    "PRICE_TREND_30D",
]


def train_and_report_v5():
    app = create_app(enable_scheduler=False)

    with app.app_context():
        df = load_and_preprocess_v5()

        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx].copy()
        test_df  = df.iloc[split_idx:].copy()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), FEATURE_COLS)
            ]
        )

        model = XGBRegressor(
            n_estimators=1200,
            learning_rate=0.015,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )

        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        print(f"[V5] 학습 시작 (데이터: {len(df):,}건, train: {len(train_df):,}, test: {len(test_df):,})")
        pipeline.fit(train_df[FEATURE_COLS], train_df["TARGET"])

        pred_log = pipeline.predict(test_df[FEATURE_COLS])
        pred     = np.expm1(pred_log)
        y_true   = np.expm1(test_df["TARGET"])

        r2  = r2_score(y_true, pred)
        mae = mean_absolute_error(y_true, pred)

        model_dir = os.path.join(project_root, "smartfarm", "ml", "models")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(pipeline, os.path.join(model_dir, "v5_tomato_price_pipeline.joblib"))

        # 피처 중요도
        importances = pipeline.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False)
        print("\n[V5] Feature Importances:")
        print(imp_df.to_string(index=False))

        # 10일 간격 예측 차이 검증 (테스트셋의 마지막 시점 기준)
        print("\n[V5] 10일 간격 예측 차이 검증 (test 마지막 100일 샘플):")
        sample = test_df.tail(100).copy()
        for offset in [0, 10, 20]:
            shifted = sample.copy()
            shifted_target = (sample["PRICE_DATE"] + pd.Timedelta(days=offset))
            t_month = shifted_target.dt.month
            t_doy   = shifted_target.dt.day_of_year
            shifted["TARGET_MONTH_SIN"] = np.sin(2 * np.pi * t_month / 12)
            shifted["TARGET_MONTH_COS"] = np.cos(2 * np.pi * t_month / 12)
            shifted["TARGET_DOY_SIN"]   = np.sin(2 * np.pi * t_doy / 365)
            shifted["TARGET_DOY_COS"]   = np.cos(2 * np.pi * t_doy / 365)
            p = np.expm1(pipeline.predict(shifted[FEATURE_COLS]))
            print(f"  +{offset:2d}일 shift → 평균 예측가: {p.mean():.0f}원  (std: {p.std():.0f}원)")

        # 시각화
        plt.figure(figsize=(15, 7))
        visual_dates = test_df["PRICE_DATE"] + pd.Timedelta(days=100)
        plt.plot(visual_dates, y_true, label="실제 가격", color='gray', alpha=0.4)
        plt.plot(visual_dates, pred,   label="V5 예측",   color='blue', linewidth=2)
        plt.title(f"토마토 수확기 가격 모델 V5 (R2: {r2:.4f}, MAE: {mae:.2f}원)")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(model_dir, "v5_price_report.png"))
        plt.close()

        print(f"\n{'='*50}")
        print(f"[V5] 학습 완료")
        print(f"R2  : {r2:.4f}  (v4 참고: 확인 필요)")
        print(f"MAE : {mae:.2f}원")
        print(f"저장 : {model_dir}/v5_tomato_price_pipeline.joblib")
        print(f"{'='*50}")


if __name__ == "__main__":
    train_and_report_v5()
