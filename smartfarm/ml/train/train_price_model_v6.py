"""
Price Model V6
- 가변 오프셋 학습: 100/125/150/180/220/260일 시나리오를 모두 학습 데이터로 포함
- DAYS_AHEAD 피처 추가: 몇 일 뒤 예측인지 모델이 인식
- 기존 v5 피처 전부 유지
- 저장: v6_tomato_price_pipeline.joblib
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

# 예측 대상 오프셋 (단기 + 장기 시나리오 전체)
TARGET_OFFSETS = [100, 125, 150, 180, 220, 260]


def load_base_data():
    print("[V6] 데이터 로드...")
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

    # 일별 연속 인덱스 (결측일 ffill)
    df = df.set_index("PRICE_DATE").resample('D').asfreq().reset_index()
    df["PRICE_PER_KG"] = df["PRICE_PER_KG"].ffill().bfill()
    df["AVG_TEMP"]     = df["AVG_TEMP"].ffill().bfill()
    return df


def build_features(df):
    """오프셋 무관한 현재 시점 피처 계산."""
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


def expand_with_offsets(df):
    """각 날짜 × 각 오프셋 = 복수의 훈련 샘플 생성."""
    frames = []
    for offset in TARGET_OFFSETS:
        tmp = df.copy()

        # 타겟: offset일 뒤 가격 (log 변환)
        tmp["TARGET"] = np.log1p(tmp["PRICE_PER_KG"].shift(-offset))

        # 오프셋 인식 피처
        tmp["DAYS_AHEAD"] = float(offset)

        # 수확 예정일 기반 계절 피처
        target_date = tmp["PRICE_DATE"] + pd.Timedelta(days=offset)
        t_month = target_date.dt.month
        t_doy   = target_date.dt.day_of_year
        tmp["TARGET_MONTH_SIN"] = np.sin(2 * np.pi * t_month / 12)
        tmp["TARGET_MONTH_COS"] = np.cos(2 * np.pi * t_month / 12)
        tmp["TARGET_DOY_SIN"]   = np.sin(2 * np.pi * t_doy / 365)
        tmp["TARGET_DOY_COS"]   = np.cos(2 * np.pi * t_doy / 365)

        frames.append(tmp)

    combined = pd.concat(frames, ignore_index=True)
    return combined.dropna(subset=["TARGET"]).reset_index(drop=True)


FEATURE_COLS = [
    # 현재 시세 상태
    "MA_30D", "MA_90D", "YOY_RATIO",
    "PRICE_LAG_90D", "PRICE_LAG_180D",
    "PRICE_MA_7D", "PRICE_TREND_30D",
    # 기상
    "GDD_30D", "TEMP_MA_30D",
    # 현재 계절
    "CURRENT_MONTH_SIN", "CURRENT_MONTH_COS",
    # 수확 예정 시점 계절
    "TARGET_MONTH_SIN", "TARGET_MONTH_COS",
    "TARGET_DOY_SIN",   "TARGET_DOY_COS",
    # ← v6 신규: 예측 오프셋
    "DAYS_AHEAD",
]


def train_and_report_v6():
    app = create_app(enable_scheduler=False)

    with app.app_context():
        base_df = load_base_data()
        feat_df = build_features(base_df)
        all_df  = expand_with_offsets(feat_df)

        print(f"[V6] 전체 샘플: {len(all_df):,}건 (날짜 {len(feat_df):,} × 오프셋 {len(TARGET_OFFSETS)}개)")
        print(f"[V6] 오프셋별 분포:")
        print(all_df.groupby("DAYS_AHEAD").size().to_string())

        # 시계열 분할: 최대 오프셋(260일)을 고려한 유효 종료일 기준 85/15
        # split 후 test 기간이 max_offset 이상 남아야 모든 오프셋 샘플이 존재함
        max_offset   = max(TARGET_OFFSETS)
        data_end     = feat_df["PRICE_DATE"].max()
        effective_end = data_end - pd.Timedelta(days=max_offset)  # 2025-09-10 근방

        valid_dates  = (
            feat_df[feat_df["PRICE_DATE"] <= effective_end]["PRICE_DATE"]
            .sort_values().unique()
        )
        split_date   = valid_dates[int(len(valid_dates) * 0.85)]

        train_df = all_df[all_df["PRICE_DATE"] <= split_date].copy()
        test_df  = all_df[
            (all_df["PRICE_DATE"] > split_date) &
            (all_df["PRICE_DATE"] <= effective_end)
        ].copy()
        print(f"\n[V6] 유효 종료일: {effective_end.date()} (data_end - {max_offset}일)")
        print(f"[V6] train: {len(train_df):,}건 | test: {len(test_df):,}건 (분할일: {split_date.date()})")

        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]), FEATURE_COLS)
        ])

        model = XGBRegressor(
            n_estimators=1200,
            learning_rate=0.015,
            max_depth=5,          # v5보다 1 깊게 — DAYS_AHEAD 분기 학습
            subsample=0.8,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )

        pipeline = Pipeline([("prep", preprocessor), ("model", model)])
        print("\n[V6] 학습 시작...")
        pipeline.fit(train_df[FEATURE_COLS], train_df["TARGET"])

        # 전체 테스트 성능
        pred_log = pipeline.predict(test_df[FEATURE_COLS])
        pred     = np.expm1(pred_log)
        y_true   = np.expm1(test_df["TARGET"])
        r2_all   = r2_score(y_true, pred)
        mae_all  = mean_absolute_error(y_true, pred)
        print(f"\n[V6] 전체 테스트  R²={r2_all:.4f}  MAE={mae_all:.2f}원")

        # 오프셋별 성능
        print("\n[V6] 오프셋별 성능:")
        for offset in TARGET_OFFSETS:
            mask = test_df["DAYS_AHEAD"] == offset
            if mask.sum() == 0:
                continue
            p = np.expm1(pipeline.predict(test_df.loc[mask, FEATURE_COLS]))
            y = np.expm1(test_df.loc[mask, "TARGET"])
            r2  = r2_score(y, p)
            mae = mean_absolute_error(y, p)
            print(f"  {offset:3d}일  R²={r2:.4f}  MAE={mae:.2f}원  (n={mask.sum()})")

        # 피처 중요도
        importances = pipeline.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False)
        print("\n[V6] Feature Importances:")
        print(imp_df.to_string(index=False))

        # 저장
        model_dir = os.path.join(project_root, "smartfarm", "ml", "models")
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, "v6_tomato_price_pipeline.joblib")
        joblib.dump(pipeline, save_path)
        print(f"\n[V6] 저장 완료: {save_path}")

        # 시각화 (오프셋별)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for ax, offset in zip(axes.flatten(), TARGET_OFFSETS):
            mask = test_df["DAYS_AHEAD"] == offset
            p = np.expm1(pipeline.predict(test_df.loc[mask, FEATURE_COLS]))
            y = np.expm1(test_df.loc[mask, "TARGET"])
            r2  = r2_score(y, p)
            mae = mean_absolute_error(y, p)
            dates = (test_df.loc[mask, "PRICE_DATE"] + pd.Timedelta(days=offset)).values
            ax.plot(dates, y.values, label="실제", color="gray",  alpha=0.5)
            ax.plot(dates, p,        label="예측", color="blue",  linewidth=1.5)
            ax.set_title(f"+{offset}일  R²={r2:.3f}  MAE={mae:.0f}원")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

        plt.suptitle("토마토 시세 모델 V6 — 오프셋별 예측 성능", fontsize=14)
        plt.tight_layout()
        report_path = os.path.join(model_dir, "v6_price_report.png")
        plt.savefig(report_path)
        plt.close()
        print(f"[V6] 리포트 저장: {report_path}")

        print(f"\n{'='*55}")
        print(f"[V6] 학습 완료")
        print(f"전체 R²  : {r2_all:.4f}")
        print(f"전체 MAE : {mae_all:.2f}원")
        print(f"피처 수  : {len(FEATURE_COLS)}개 (DAYS_AHEAD 추가)")
        print(f"{'='*55}")


if __name__ == "__main__":
    train_and_report_v6()
