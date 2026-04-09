import os
import sys
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------
# 경로
# --------------------------------------------------
def get_model_dir():
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


# --------------------------------------------------
# 유틸
# --------------------------------------------------
def first_valid(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) > 0 else np.nan


def last_valid(series: pd.Series):
    s = series.dropna()
    return s.iloc[-1] if len(s) > 0 else np.nan


# --------------------------------------------------
# 데이터 로드
# --------------------------------------------------
def load_raw_data():
    """
    사용 데이터
    - GROW_SUMMARY: ORIGIN_TYPE = 0, PLANT_NUM 있는 생육 데이터
    - PRODUCTS: 첫 수확일 계산
    - CULTIVATIONS: planting 정보
    """
    grow_query = text("""
        SELECT
            G.CULT_ID,
            G.PLANT_NUM,
            G.INSPECT_DATE,
            G.GROWTH_DAYS AS DAP,
            G.PLANT_HEIGHT,
            G.GROWTH_LENGTH,
            G.LEAF_COUNT,
            G.LEAF_LENGTH,
            G.LEAF_WIDTH,
            G.BRANCH_WIDTH,
            G.CLUSTER_HEIGHT,
            G.CLUSTER_NUM,
            G.FLOWERS_PER_CLUSTER,
            G.BLOOMING_PER_CLUSTER,
            G.FRUITS_PER_CLUSTER
        FROM GROW_SUMMARY G
        WHERE G.ORIGIN_TYPE = 0
          AND G.PLANT_NUM IS NOT NULL
          AND G.GROWTH_DAYS BETWEEN 0 AND 350
    """)

    prod_query = text("""
        SELECT
            P.CULT_ID,
            P.PRODUCTION_DATE
        FROM PRODUCTS P
        WHERE P.PRODUCTION_DATE IS NOT NULL
    """)

    cult_query = text("""
        SELECT
            C.CULT_ID,
            C.PLANTING_DATE,
            C.CROP_CYCLE,
            C.ITEM,
            C.ITEM_VARIETY,
            C.HOUSE_TYPE,
            C.HOUSE_FORM,
            C.PLANTING_AREA,
            C.PLANTING_DENSITY
        FROM CULTIVATIONS C
    """)

    df_grow = pd.read_sql(grow_query, db.engine)
    df_prod = pd.read_sql(prod_query, db.engine)
    df_cult = pd.read_sql(cult_query, db.engine)

    df_grow.columns = [c.upper() for c in df_grow.columns]
    df_prod.columns = [c.upper() for c in df_prod.columns]
    df_cult.columns = [c.upper() for c in df_cult.columns]

    df_grow["INSPECT_DATE"] = pd.to_datetime(df_grow["INSPECT_DATE"], errors="coerce")
    df_prod["PRODUCTION_DATE"] = pd.to_datetime(df_prod["PRODUCTION_DATE"], errors="coerce")
    df_cult["PLANTING_DATE"] = pd.to_datetime(df_cult["PLANTING_DATE"], errors="coerce")

    numeric_cols = [
        "DAP",
        "PLANT_HEIGHT",
        "GROWTH_LENGTH",
        "LEAF_COUNT",
        "LEAF_LENGTH",
        "LEAF_WIDTH",
        "BRANCH_WIDTH",
        "CLUSTER_HEIGHT",
        "CLUSTER_NUM",
        "FLOWERS_PER_CLUSTER",
        "BLOOMING_PER_CLUSTER",
        "FRUITS_PER_CLUSTER",
        "PLANTING_AREA",
        "PLANTING_DENSITY",
        "CROP_CYCLE",
    ]

    for col in numeric_cols:
        if col in df_grow.columns:
            df_grow[col] = pd.to_numeric(df_grow[col], errors="coerce")
        if col in df_cult.columns:
            df_cult[col] = pd.to_numeric(df_cult[col], errors="coerce")

    return df_grow, df_prod, df_cult


# --------------------------------------------------
# 첫 수확일 생성
# --------------------------------------------------
def build_first_harvest_table(df_prod: pd.DataFrame, df_cult: pd.DataFrame):
    """
    CULT_ID별 첫 수확일 / 첫 수확 DAP
    """
    first_prod = (
        df_prod.groupby("CULT_ID", as_index=False)
        .agg(FIRST_HARVEST_DATE=("PRODUCTION_DATE", "min"))
    )

    df_first = first_prod.merge(df_cult, on="CULT_ID", how="inner")
    df_first["FIRST_HARVEST_DAP"] = (
        df_first["FIRST_HARVEST_DATE"] - df_first["PLANTING_DATE"]
    ).dt.days

    df_first = df_first[df_first["FIRST_HARVEST_DAP"].notna()].copy()
    df_first = df_first[df_first["FIRST_HARVEST_DAP"] >= 0].copy()

    return df_first


# --------------------------------------------------
# 첫 수확 전 생육만 남기기
# --------------------------------------------------
def filter_pre_harvest_growth(df_grow: pd.DataFrame, df_first: pd.DataFrame):
    """
    첫 수확일 이전 생육 데이터만 사용
    """
    df = df_grow.merge(
        df_first[["CULT_ID", "FIRST_HARVEST_DATE", "FIRST_HARVEST_DAP"]],
        on="CULT_ID",
        how="inner"
    )

    df = df[df["INSPECT_DATE"].notna()].copy()
    df = df[df["INSPECT_DATE"] < df["FIRST_HARVEST_DATE"]].copy()
    df = df[df["DAP"].notna()].copy()
    df = df[df["DAP"] >= 0].copy()

    return df


# --------------------------------------------------
# 개체별 pre-harvest 생육기간 요약
# --------------------------------------------------
def summarize_preharvest_growth_by_plant(df_pre: pd.DataFrame):
    growth_cols = [
        "PLANT_HEIGHT",
        "GROWTH_LENGTH",
        "LEAF_COUNT",
        "CLUSTER_NUM",
        "FLOWERS_PER_CLUSTER",
        "FRUITS_PER_CLUSTER",
    ]

    rows = []

    for (cult_id, plant_num), g in df_pre.groupby(["CULT_ID", "PLANT_NUM"]):
        g = g.sort_values(["INSPECT_DATE", "DAP"]).reset_index(drop=True)

        if len(g) < 2:
            continue

        first_date = g["INSPECT_DATE"].iloc[0]
        last_date = g["INSPECT_DATE"].iloc[-1]
        first_dap = g["DAP"].iloc[0]
        last_dap = g["DAP"].iloc[-1]
        obs_count = len(g)
        period_days = last_dap - first_dap

        if pd.isna(period_days) or period_days <= 0:
            continue

        row = {
            "CULT_ID": cult_id,
            "PLANT_NUM": plant_num,
            "OBS_COUNT": obs_count,
            "FIRST_INSPECT_DATE": first_date,
            "LAST_INSPECT_DATE": last_date,
            "FIRST_DAP": first_dap,
            "LAST_DAP": last_dap,
            "PREHARVEST_GROWTH_PERIOD_DAYS": period_days,
        }

        for col in growth_cols:
            first_val = first_valid(g[col])
            last_val = last_valid(g[col])

            row[f"FIRST_{col}"] = first_val
            row[f"LAST_{col}"] = last_val

            if pd.notna(first_val) and pd.notna(last_val):
                delta_val = last_val - first_val
            else:
                delta_val = np.nan

            row[f"DELTA_{col}"] = delta_val

            if pd.notna(delta_val) and period_days > 0:
                row[f"SPEED_{col}"] = delta_val / period_days
            else:
                row[f"SPEED_{col}"] = np.nan

        rows.append(row)

    df_plant = pd.DataFrame(rows)

    if df_plant.empty:
        return df_plant

    return df_plant.sort_values(["CULT_ID", "PLANT_NUM"]).reset_index(drop=True)


# --------------------------------------------------
# 개체 요약을 CULT_ID 단위 평균으로 변환
# --------------------------------------------------
def aggregate_plant_summary_to_cult(df_plant: pd.DataFrame):
    if df_plant.empty:
        return pd.DataFrame()

    numeric_cols = [
        c for c in df_plant.columns
        if c not in ["CULT_ID", "PLANT_NUM", "FIRST_INSPECT_DATE", "LAST_INSPECT_DATE"]
    ]

    agg_dict = {col: "mean" for col in numeric_cols}
    agg_dict["PLANT_NUM"] = "nunique"

    df_cult_growth = (
        df_plant.groupby("CULT_ID", as_index=False)
        .agg(agg_dict)
        .rename(columns={"PLANT_NUM": "PLANT_CNT"})
        .sort_values("CULT_ID")
        .reset_index(drop=True)
    )

    return df_cult_growth


# --------------------------------------------------
# 학습용 최종 테이블 생성
# --------------------------------------------------
def build_training_dataset(df_cult_growth: pd.DataFrame, df_first: pd.DataFrame):
    if df_cult_growth.empty:
        return pd.DataFrame()

    df = df_cult_growth.merge(df_first, on="CULT_ID", how="inner")

    # 현재 시점 = 마지막 pre-harvest 관측 시점
    df["DAYS_TO_FIRST_HARVEST"] = df["FIRST_HARVEST_DAP"] - df["LAST_DAP"]

    # 비정상 제거
    df = df[df["DAYS_TO_FIRST_HARVEST"].notna()].copy()
    df = df[df["DAYS_TO_FIRST_HARVEST"] >= 0].copy()

    return df


# --------------------------------------------------
# 저장
# --------------------------------------------------
def save_feature_importance_chart(importances, score, mae, train_rows, test_rows, model_dir):
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.barh(importances.index, importances.values)

    ax.set_title("수확개시시점 예측 모델 중요도", fontsize=14)
    ax.set_xlabel("Importance Score")

    max_val = importances.max() if len(importances) > 0 else 0
    offset = max(max_val * 0.01, 0.001)

    for bar, value in zip(bars, importances.values):
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=9
        )

    summary_text = (
        f"R2 Score: {score:.4f}\n"
        f"MAE: {mae:.2f} days\n"
        f"Train Rows: {train_rows:,}\n"
        f"Test Rows: {test_rows:,}\n"
        f"Feature Count: {len(importances)}"
    )

    fig.text(
        0.74,
        0.18,
        summary_text,
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray"
        )
    )

    plt.tight_layout(rect=[0, 0, 0.92, 1])

    img_path = os.path.join(model_dir, "harvest_start_model_importance.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    print(f"[이미지 저장 완료]: {img_path}")
    plt.show()

    return img_path


def save_model(model_data, model_dir):
    model_path = os.path.join(model_dir, "harvest_start_model.joblib")
    joblib.dump(model_data, model_path)
    print(f"[모델 저장 완료]: {model_path}")
    return model_path


# --------------------------------------------------
# 메인
# --------------------------------------------------
def run_harvest_start_model():
    try:
        app = create_app(enable_scheduler=False)
    except TypeError:
        app = create_app()

    with app.app_context():
        print("1. 원본 데이터 로드...")
        df_grow, df_prod, df_cult = load_raw_data()
        print(f" - GROW_SUMMARY rows: {len(df_grow):,}")
        print(f" - PRODUCTS rows: {len(df_prod):,}")
        print(f" - CULTIVATIONS rows: {len(df_cult):,}")

        print("\n2. 첫 수확일 생성...")
        df_first = build_first_harvest_table(df_prod, df_cult)
        print(f" - first harvest rows: {len(df_first):,}")

        print("\n3. 첫 수확 전 생육만 필터...")
        df_pre = filter_pre_harvest_growth(df_grow, df_first)
        print(f" - pre-harvest growth rows: {len(df_pre):,}")

        print("\n4. 개체별 pre-harvest 생육기간 요약...")
        df_plant = summarize_preharvest_growth_by_plant(df_pre)
        print(f" - plant summary rows: {len(df_plant):,}")

        print("\n5. CULT_ID 단위 평균 집계...")
        df_cult_growth = aggregate_plant_summary_to_cult(df_plant)
        print(f" - cult growth rows: {len(df_cult_growth):,}")

        print("\n6. 학습용 최종 데이터 생성...")
        df_final = build_training_dataset(df_cult_growth, df_first)
        print(f" - final rows: {len(df_final):,}")

        if df_final.empty:
            print("학습용 데이터가 없습니다.")
            return

        feature_cols = [
            # 기본 관측 정보
            "PLANT_CNT",
            "OBS_COUNT",
            "LAST_DAP",
            "PREHARVEST_GROWTH_PERIOD_DAYS",

            # 현재 상태
            "LAST_PLANT_HEIGHT",
            "LAST_GROWTH_LENGTH",
            "LAST_LEAF_COUNT",
            "LAST_CLUSTER_NUM",
            "LAST_FLOWERS_PER_CLUSTER",
            "LAST_FRUITS_PER_CLUSTER",

            # 최근 변화량
            "DELTA_PLANT_HEIGHT",
            "DELTA_GROWTH_LENGTH",
            "DELTA_LEAF_COUNT",
            "DELTA_CLUSTER_NUM",
            "DELTA_FLOWERS_PER_CLUSTER",
            "DELTA_FRUITS_PER_CLUSTER",

            # 최근 변화속도
            "SPEED_PLANT_HEIGHT",
            "SPEED_GROWTH_LENGTH",
            "SPEED_LEAF_COUNT",
            "SPEED_CLUSTER_NUM",
            "SPEED_FLOWERS_PER_CLUSTER",
            "SPEED_FRUITS_PER_CLUSTER",
        ]

        df_ml = df_final.dropna(subset=feature_cols + ["DAYS_TO_FIRST_HARVEST"]).copy()

        # 최소 관측 수
        df_ml = df_ml[df_ml["OBS_COUNT"] >= 2].copy()

        # 변화속도 계산 가능한 경우만
        df_ml = df_ml[df_ml["PREHARVEST_GROWTH_PERIOD_DAYS"] > 0].copy()

        # 이상치 제거
        df_ml = df_ml[df_ml["DAYS_TO_FIRST_HARVEST"] <= 20].copy()

        print(f" - 학습 가능 rows: {len(df_ml):,}")
        print("\n[DAYS_TO_FIRST_HARVEST 분포]")
        print(df_ml["DAYS_TO_FIRST_HARVEST"].describe())

        if len(df_ml) < 30:
            print("학습 가능 데이터가 너무 적습니다.")
            return

        X = df_ml[feature_cols]
        y = df_ml["DAYS_TO_FIRST_HARVEST"]
        groups = df_ml["CULT_ID"]

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=42
        )

        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        print("\n7. 모델 학습...")
        print(f" - 학습 CULT_ID 수: {df_ml.iloc[train_idx]['CULT_ID'].nunique()}")
        print(f" - 평가 CULT_ID 수: {df_ml.iloc[test_idx]['CULT_ID'].nunique()}")
        print(f" - 학습 row 수: {len(X_train):,}")
        print(f" - 평가 row 수: {len(X_test):,}")

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred = model.predict(X_test)

        score = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print("-" * 50)
        print(f"R2 Score: {score:.4f}")
        print(f"MAE: {mae:.2f} days")

        importances = pd.Series(
            model.feature_importances_,
            index=feature_cols
        ).sort_values(ascending=False)

        print("\n상위 중요도")
        print(importances)

        model_dir = get_model_dir()

        img_path = save_feature_importance_chart(
            importances=importances,
            score=score,
            mae=mae,
            train_rows=len(X_train),
            test_rows=len(X_test),
            model_dir=model_dir
        )

        model_data = {
            "model": model,
            "features": feature_cols,
            "target_col": "DAYS_TO_FIRST_HARVEST",
            "target_desc": "첫 수확까지 남은 일수",
            "r2_score": float(score),
            "mae": float(mae),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_importances": {k: float(v) for k, v in importances.items()},
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        model_path = save_model(model_data, model_dir)

        debug_cols = [
            "CULT_ID",
            "PLANT_CNT",
            "OBS_COUNT",
            "LAST_DAP",
            "PREHARVEST_GROWTH_PERIOD_DAYS",
            "FIRST_HARVEST_DAP",
            "DAYS_TO_FIRST_HARVEST",

            "LAST_PLANT_HEIGHT",
            "LAST_GROWTH_LENGTH",
            "LAST_LEAF_COUNT",
            "LAST_CLUSTER_NUM",
            "LAST_FLOWERS_PER_CLUSTER",
            "LAST_FRUITS_PER_CLUSTER",

            "DELTA_PLANT_HEIGHT",
            "DELTA_GROWTH_LENGTH",
            "DELTA_LEAF_COUNT",
            "DELTA_CLUSTER_NUM",
            "DELTA_FLOWERS_PER_CLUSTER",
            "DELTA_FRUITS_PER_CLUSTER",

            "SPEED_PLANT_HEIGHT",
            "SPEED_GROWTH_LENGTH",
            "SPEED_LEAF_COUNT",
            "SPEED_CLUSTER_NUM",
            "SPEED_FLOWERS_PER_CLUSTER",
            "SPEED_FRUITS_PER_CLUSTER",
        ]

        debug_path = os.path.join(model_dir, "harvest_start_model_debug.csv")
        df_ml[debug_cols].to_csv(debug_path, index=False, encoding="utf-8-sig")
        print(f"[디버그 CSV 저장 완료]: {debug_path}")

        print("\n8. 최종 결과")
        print(f" - model_path: {model_path}")
        print(f" - img_path: {img_path}")
        print(f" - debug_path: {debug_path}")


if __name__ == "__main__":
    run_harvest_start_model()