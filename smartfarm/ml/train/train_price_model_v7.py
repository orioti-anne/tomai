"""
Price Model V7
- 오프셋별 개별 LightGBM 모델 (6개)
- 단기: 100/125/150일 | 장기: 180/220/260일
- DAYS_AHEAD 불필요 (각 모델이 오프셋 전담)
- 저장: v7_tomato_price_pipelines.joblib  {offset: pipeline}
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sqlalchemy import text
import lightgbm as lgb
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

TARGET_OFFSETS = [100, 125, 150, 180, 220, 260]

FEATURE_COLS = [
    "MA_30D", "MA_90D", "YOY_RATIO",
    "PRICE_LAG_90D", "PRICE_LAG_180D",
    "PRICE_MA_7D", "PRICE_TREND_30D",
    "GDD_30D", "TEMP_MA_30D",
    "CURRENT_MONTH_SIN", "CURRENT_MONTH_COS",
    "TARGET_MONTH_SIN", "TARGET_MONTH_COS",
    "TARGET_DOY_SIN",   "TARGET_DOY_COS",
]


def load_base_data():
    print("[V7] 데이터 로드...")
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
    df["AVG_TEMP"]     = df["AVG_TEMP"].ffill().bfill()
    return df


def build_features(df):
    df = df.copy()
    df["MA_30D"]       = df["PRICE_PER_KG"].shift(1).rolling(30).mean()
    df["MA_90D"]       = df["PRICE_PER_KG"].shift(1).rolling(90).mean()
    prev_year          = df["PRICE_PER_KG"].shift(365)
    df["YOY_RATIO"]    = (df["PRICE_PER_KG"].shift(1) / prev_year).clip(0.5, 1.5).fillna(1.0)
    df["TEMP_MA_30D"]  = df["AVG_TEMP"].shift(1).rolling(30).mean()
    df["GDD_30D"]      = (df["AVG_TEMP"] - 10).clip(lower=0).shift(1).rolling(30).sum()
    df["PRICE_LAG_90D"]  = df["PRICE_PER_KG"].shift(90)
    df["PRICE_LAG_180D"] = df["PRICE_PER_KG"].shift(180)
    curr_month = df["PRICE_DATE"].dt.month
    df["CURRENT_MONTH_SIN"] = np.sin(2 * np.pi * curr_month / 12)
    df["CURRENT_MONTH_COS"] = np.cos(2 * np.pi * curr_month / 12)
    df["PRICE_MA_7D"]      = df["PRICE_PER_KG"].shift(1).rolling(7).mean()
    df["PRICE_TREND_30D"]  = (df["MA_30D"] - df["MA_90D"]) / df["MA_90D"].replace(0, np.nan)
    return df.dropna(subset=["MA_90D", "PRICE_LAG_180D"]).reset_index(drop=True)


def make_offset_df(feat_df, offset):
    """단일 오프셋에 대한 target + 계절 피처 생성."""
    tmp = feat_df.copy()
    tmp["TARGET"] = np.log1p(tmp["PRICE_PER_KG"].shift(-offset))
    target_date = tmp["PRICE_DATE"] + pd.Timedelta(days=offset)
    t_month = target_date.dt.month
    t_doy   = target_date.dt.day_of_year
    tmp["TARGET_MONTH_SIN"] = np.sin(2 * np.pi * t_month / 12)
    tmp["TARGET_MONTH_COS"] = np.cos(2 * np.pi * t_month / 12)
    tmp["TARGET_DOY_SIN"]   = np.sin(2 * np.pi * t_doy / 365)
    tmp["TARGET_DOY_COS"]   = np.cos(2 * np.pi * t_doy / 365)
    return tmp.dropna(subset=["TARGET"]).reset_index(drop=True)


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), FEATURE_COLS)
    ])


def make_lgbm(offset):
    """오프셋에 따라 파라미터 미세 조정."""
    if offset <= 150:
        # 단기: 더 많은 트리, 높은 학습률
        return lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        # 장기: 복잡도 낮춰 과적합 방지
        return lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.008,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )


def train_and_report_v7():
    app = create_app(enable_scheduler=False)

    with app.app_context():
        base_df = load_base_data()
        feat_df = build_features(base_df)

        # split 기준: data_end - max_offset → 모든 오프셋이 test 샘플 보유 보장
        max_offset    = max(TARGET_OFFSETS)
        data_end      = feat_df["PRICE_DATE"].max()
        effective_end = data_end - pd.Timedelta(days=max_offset)
        valid_dates   = (
            feat_df[feat_df["PRICE_DATE"] <= effective_end]["PRICE_DATE"]
            .sort_values().unique()
        )
        split_date = valid_dates[int(len(valid_dates) * 0.85)]
        print(f"[V7] 유효 종료일: {effective_end.date()} | 분할일: {split_date.date()}")

        pipelines   = {}
        results_tbl = []

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for ax, offset in zip(axes.flatten(), TARGET_OFFSETS):
            print(f"\n{'─'*45}")
            print(f"[V7] +{offset}일 모델 학습...")

            df_off = make_offset_df(feat_df, offset)

            train_df = df_off[df_off["PRICE_DATE"] <= split_date].copy()
            test_df  = df_off[
                (df_off["PRICE_DATE"] > split_date) &
                (df_off["PRICE_DATE"] <= effective_end)
            ].copy()
            print(f"     train={len(train_df):,}건 | test={len(test_df):,}건")

            pipeline = Pipeline([
                ("prep",  make_preprocessor()),
                ("model", make_lgbm(offset)),
            ])
            pipeline.fit(train_df[FEATURE_COLS], train_df["TARGET"])

            pred_log = pipeline.predict(test_df[FEATURE_COLS])
            pred     = np.expm1(pred_log)
            y_true   = np.expm1(test_df["TARGET"])

            r2  = r2_score(y_true, pred)
            mae = mean_absolute_error(y_true, pred)
            print(f"     R²={r2:.4f}  MAE={mae:.2f}원")

            pipelines[offset] = pipeline
            results_tbl.append({"offset": offset, "r2": r2, "mae": mae, "n_test": len(test_df)})

            dates = (test_df["PRICE_DATE"] + pd.Timedelta(days=offset)).values
            ax.plot(dates, y_true.values, label="실제", color="gray",  alpha=0.5)
            ax.plot(dates, pred,          label="예측", color="steelblue", linewidth=1.5)
            ax.set_title(f"+{offset}일  R²={r2:.3f}  MAE={mae:.0f}원")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

        # 저장
        model_dir = os.path.join(project_root, "smartfarm", "ml", "models")
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, "v7_tomato_price_pipelines.joblib")
        joblib.dump(pipelines, save_path)
        print(f"\n[V7] 저장 완료: {save_path}")

        # 시각화 저장
        plt.suptitle("토마토 시세 모델 V7 (LightGBM 오프셋별) — 예측 성능", fontsize=14)
        plt.tight_layout()
        report_path = os.path.join(model_dir, "v7_price_report.png")
        plt.savefig(report_path)
        plt.close()
        print(f"[V7] 리포트 저장: {report_path}")

        # 요약
        print(f"\n{'='*55}")
        print(f"[V7] 학습 완료 요약")
        print(f"{'오프셋':>6}  {'R²':>8}  {'MAE':>10}  {'n_test':>7}")
        print(f"{'─'*45}")
        total_r2  = np.mean([r["r2"]  for r in results_tbl])
        total_mae = np.mean([r["mae"] for r in results_tbl])
        for r in results_tbl:
            print(f"  {r['offset']:3d}일  R²={r['r2']:7.4f}  MAE={r['mae']:8.2f}원  n={r['n_test']}")
        print(f"{'─'*45}")
        print(f"  평균    R²={total_r2:7.4f}  MAE={total_mae:8.2f}원")
        print(f"{'='*55}")


if __name__ == "__main__":
    train_and_report_v7()
