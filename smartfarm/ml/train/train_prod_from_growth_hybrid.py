# train_prod_from_growth_plant_summary.py

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
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import create_app, db

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------
# 공통
# --------------------------------------------------
def get_model_dir():
    model_dir = os.path.join(project_root, "ml", "models")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def first_valid(series):
    s = series.dropna()
    return s.iloc[0] if len(s) > 0 else np.nan


def last_valid(series):
    s = series.dropna()
    return s.iloc[-1] if len(s) > 0 else np.nan


# --------------------------------------------------
# 데이터 로드
# --------------------------------------------------
def load_raw_data():
    """
    - GROW_SUMMARY: ORIGIN_TYPE = 0만 사용
    - PRODUCTS: 원본 판매행
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
            P.PRODUCTION_DATE,
            P.TOTAL_QUANTITY,
            P.TOTAL_SALES
        FROM PRODUCTS P
    """)

    df_grow = pd.read_sql(grow_query, db.engine)
    df_prod = pd.read_sql(prod_query, db.engine)

    df_grow.columns = [c.upper() for c in df_grow.columns]
    df_prod.columns = [c.upper() for c in df_prod.columns]

    df_grow["INSPECT_DATE"] = pd.to_datetime(df_grow["INSPECT_DATE"])
    df_prod["PRODUCTION_DATE"] = pd.to_datetime(df_prod["PRODUCTION_DATE"])

    return df_grow, df_prod


# --------------------------------------------------
# 판매 데이터 집계
# --------------------------------------------------
def aggregate_products_daily(df_prod):
    """
    같은 CULT_ID + PRODUCTION_DATE의 판매량/판매금액 합산
    """
    prod_daily = (
        df_prod.groupby(["CULT_ID", "PRODUCTION_DATE"], as_index=False)
        .agg({
            "TOTAL_QUANTITY": "sum",
            "TOTAL_SALES": "sum"
        })
        .rename(columns={
            "TOTAL_QUANTITY": "DAILY_QTY",
            "TOTAL_SALES": "DAILY_SALES"
        })
        .sort_values(["CULT_ID", "PRODUCTION_DATE"])
        .reset_index(drop=True)
    )
    return prod_daily


def aggregate_products_total(prod_daily):
    """
    CULT_ID 전체 판매량/판매금액
    """
    prod_total = (
        prod_daily.groupby("CULT_ID", as_index=False)
        .agg({
            "DAILY_QTY": "sum",
            "DAILY_SALES": "sum"
        })
        .rename(columns={
            "DAILY_QTY": "CULT_TOTAL_QTY",
            "DAILY_SALES": "CULT_TOTAL_SALES"
        })
    )
    return prod_total


# --------------------------------------------------
# 개체별 전체 생육기간 요약
# --------------------------------------------------
def summarize_growth_by_plant(df_grow):
    """
    CULT_ID + PLANT_NUM 기준으로
    - 관측기간
    - 마지막 상태
    - 증가량
    - 평균 속도
    생성
    """
    growth_cols = [
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
        "FRUITS_PER_CLUSTER"
    ]

    rows = []

    for (cult_id, plant_num), g in df_grow.groupby(["CULT_ID", "PLANT_NUM"]):
        g = g.sort_values(["INSPECT_DATE", "DAP"]).reset_index(drop=True)

        obs_count = len(g)
        first_inspect_date = g["INSPECT_DATE"].iloc[0]
        last_inspect_date = g["INSPECT_DATE"].iloc[-1]
        first_dap = g["DAP"].iloc[0]
        last_dap = g["DAP"].iloc[-1]
        obs_period_days = last_dap - first_dap

        row = {
            "CULT_ID": cult_id,
            "PLANT_NUM": plant_num,
            "OBS_COUNT": obs_count,
            "FIRST_INSPECT_DATE": first_inspect_date,
            "LAST_INSPECT_DATE": last_inspect_date,
            "FIRST_DAP": first_dap,
            "LAST_DAP": last_dap,
            "OBS_PERIOD_DAYS": obs_period_days
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

            if pd.notna(delta_val) and pd.notna(obs_period_days) and obs_period_days > 0:
                row[f"SPEED_{col}"] = delta_val / obs_period_days
            else:
                row[f"SPEED_{col}"] = np.nan

        rows.append(row)

    df_summary = pd.DataFrame(rows)

    return df_summary.sort_values(["CULT_ID", "PLANT_NUM"]).reset_index(drop=True)


# --------------------------------------------------
# target 생성
# --------------------------------------------------
def add_targets(df_plant, prod_daily, prod_total):
    """
    target 1: 개체 마지막 조사일 이후 7일 내 총 판매량
    target 2: 해당 CULT_ID 총 판매량
    """
    daily_grouped = {
        cult_id: g.sort_values("PRODUCTION_DATE")
        for cult_id, g in prod_daily.groupby("CULT_ID")
    }

    total_qty_map = prod_total.set_index("CULT_ID")["CULT_TOTAL_QTY"].to_dict()
    total_sales_map = prod_total.set_index("CULT_ID")["CULT_TOTAL_SALES"].to_dict()

    next_7d_qty = []
    next_7d_sales = []
    next_7d_event_count = []

    for _, row in df_plant.iterrows():
        cult_id = row["CULT_ID"]
        last_date = row["LAST_INSPECT_DATE"]

        if cult_id not in daily_grouped:
            next_7d_qty.append(0.0)
            next_7d_sales.append(0.0)
            next_7d_event_count.append(0)
            continue

        g = daily_grouped[cult_id]

        future_7d = g[
            (g["PRODUCTION_DATE"] > last_date) &
            (g["PRODUCTION_DATE"] <= last_date + pd.Timedelta(days=7))
        ]

        next_7d_qty.append(float(future_7d["DAILY_QTY"].sum()))
        next_7d_sales.append(float(future_7d["DAILY_SALES"].sum()))
        next_7d_event_count.append(int(len(future_7d)))

    df = df_plant.copy()
    df["TARGET_QTY_7D_AFTER_LAST"] = next_7d_qty
    df["TARGET_SALES_7D_AFTER_LAST"] = next_7d_sales
    df["TARGET_EVENT_CNT_7D_AFTER_LAST"] = next_7d_event_count

    df["TARGET_QTY_TOTAL_CULT"] = df["CULT_ID"].map(total_qty_map)
    df["TARGET_SALES_TOTAL_CULT"] = df["CULT_ID"].map(total_sales_map)

    return df


# --------------------------------------------------
# 저장
# --------------------------------------------------
def save_importance_chart(
    importances,
    title,
    score,
    mae,
    train_rows,
    test_rows,
    output_png_path
):
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(15, 10))
    bars = ax.barh(importances.index, importances.values)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Importance Score")

    max_val = importances.max() if len(importances) > 0 else 0
    offset = max_val * 0.01 if max_val > 0 else 0.001

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
        f"MAE: {mae:.2f}\n"
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
    plt.savefig(output_png_path, dpi=150, bbox_inches="tight")
    print(f"[이미지 저장 완료]: {output_png_path}")
    plt.show()


def save_model(model_data, output_joblib_path):
    joblib.dump(model_data, output_joblib_path)
    print(f"[모델 저장 완료]: {output_joblib_path}")


# --------------------------------------------------
# 모델 학습
# --------------------------------------------------
def train_one_model(df_ml, feature_cols, target_col, model_name_prefix):
    X = df_ml[feature_cols]
    y = df_ml[target_col]
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

    print(f"\n[{target_col}] 모델 학습 시작...")
    print(f" - 학습 CULT_ID 수: {df_ml.iloc[train_idx]['CULT_ID'].nunique()}")
    print(f" - 평가 CULT_ID 수: {df_ml.iloc[test_idx]['CULT_ID'].nunique()}")
    print(f" - 학습 row 수: {len(X_train):,}")
    print(f" - 평가 row 수: {len(X_test):,}")

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=1.0,
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
    print(f"MAE: {mae:.2f}")

    importances = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print("\n상위 중요도 15개")
    print(importances.head(15))

    model_dir = get_model_dir()

    joblib_path = os.path.join(model_dir, f"{model_name_prefix}.joblib")
    png_path = os.path.join(model_dir, f"{model_name_prefix}.png")

    model_data = {
        "model": model,
        "features": feature_cols,
        "target_col": target_col,
        "r2_score": float(score),
        "mae": float(mae),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_importances": {k: float(v) for k, v in importances.items()},
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    save_model(model_data, joblib_path)

    save_importance_chart(
        importances=importances,
        title=f"{target_col} 중요도",
        score=score,
        mae=mae,
        train_rows=len(X_train),
        test_rows=len(X_test),
        output_png_path=png_path
    )

    return {
        "target_col": target_col,
        "r2_score": score,
        "mae": mae,
        "joblib_path": joblib_path,
        "png_path": png_path
    }


# --------------------------------------------------
# 메인
# --------------------------------------------------
def run_model():
    try:
        app = create_app(enable_scheduler=False)
    except TypeError:
        app = create_app()

    with app.app_context():
        print("1. 원본 데이터 로드...")
        df_grow, df_prod = load_raw_data()

        print(f" - GROW_SUMMARY rows: {len(df_grow):,}")
        print(f" - PRODUCTS rows: {len(df_prod):,}")

        print("\n2. 판매 일자 집계...")
        prod_daily = aggregate_products_daily(df_prod)
        prod_total = aggregate_products_total(prod_daily)

        print(f" - prod_daily rows: {len(prod_daily):,}")
        print(f" - prod_total cult_id 수: {len(prod_total):,}")

        print("\n3. 개체별 전체 생육기간 요약...")
        df_plant = summarize_growth_by_plant(df_grow)
        print(f" - plant summary rows: {len(df_plant):,}")

        print("\n4. target 생성...")
        df_final = add_targets(df_plant, prod_daily, prod_total)

        print("\n5. target 분포 확인...")
        print("\n[TARGET_QTY_7D_AFTER_LAST]")
        print(df_final["TARGET_QTY_7D_AFTER_LAST"].describe())

        print("\n[TARGET_QTY_TOTAL_CULT]")
        print(df_final["TARGET_QTY_TOTAL_CULT"].describe())

        # 개체별 전체 생육기간 기반 feature
        feature_cols = [
            "OBS_COUNT",
            "FIRST_DAP",
            "LAST_DAP",
            "OBS_PERIOD_DAYS",

            "LAST_PLANT_HEIGHT",
            "LAST_GROWTH_LENGTH",
            "LAST_LEAF_COUNT",
            "LAST_LEAF_LENGTH",
            "LAST_LEAF_WIDTH",
            "LAST_BRANCH_WIDTH",
            "LAST_CLUSTER_HEIGHT",
            "LAST_CLUSTER_NUM",
            "LAST_FLOWERS_PER_CLUSTER",
            "LAST_BLOOMING_PER_CLUSTER",
            "LAST_FRUITS_PER_CLUSTER",

            "DELTA_PLANT_HEIGHT",
            "DELTA_GROWTH_LENGTH",
            "DELTA_LEAF_COUNT",
            "DELTA_LEAF_LENGTH",
            "DELTA_LEAF_WIDTH",
            "DELTA_BRANCH_WIDTH",
            "DELTA_CLUSTER_HEIGHT",
            "DELTA_CLUSTER_NUM",
            "DELTA_FLOWERS_PER_CLUSTER",
            "DELTA_BLOOMING_PER_CLUSTER",
            "DELTA_FRUITS_PER_CLUSTER",

            "SPEED_PLANT_HEIGHT",
            "SPEED_GROWTH_LENGTH",
            "SPEED_LEAF_COUNT",
            "SPEED_LEAF_LENGTH",
            "SPEED_LEAF_WIDTH",
            "SPEED_BRANCH_WIDTH",
            "SPEED_CLUSTER_HEIGHT",
            "SPEED_CLUSTER_NUM",
            "SPEED_FLOWERS_PER_CLUSTER",
            "SPEED_BLOOMING_PER_CLUSTER",
            "SPEED_FRUITS_PER_CLUSTER"
        ]

        print("\n6. 학습용 데이터 생성...")

        # 관측이 2회 이상 있어야 증가량/속도 의미가 있음
        df_base = df_final[df_final["OBS_COUNT"] >= 2].copy()

        # 모델 1: 마지막 조사일 이후 7일 판매량
        df_ml_7d = df_base.dropna(
            subset=feature_cols + ["TARGET_QTY_7D_AFTER_LAST"]
        ).copy()

        print(f" - 7일 target 학습 rows: {len(df_ml_7d):,}")

        # 모델 2: cult 전체 판매량
        df_ml_total = df_base.dropna(
            subset=feature_cols + ["TARGET_QTY_TOTAL_CULT"]
        ).copy()

        print(f" - 총판매량 target 학습 rows: {len(df_ml_total):,}")

        print("\n7. 최근 50행 디버그 저장...")
        debug_cols = [
            "CULT_ID",
            "PLANT_NUM",
            "FIRST_INSPECT_DATE",
            "LAST_INSPECT_DATE",
            "FIRST_DAP",
            "LAST_DAP",
            "OBS_PERIOD_DAYS",
            "TARGET_QTY_7D_AFTER_LAST",
            "TARGET_QTY_TOTAL_CULT"
        ]

        debug_path = os.path.join(get_model_dir(), "prod_growth_plant_summary_debug.csv")
        df_final.tail(50)[debug_cols].to_csv(debug_path, index=False, encoding="utf-8-sig")
        print(f"[디버그 CSV 저장 완료]: {debug_path}")

        results = []

        if len(df_ml_7d) > 0:
            results.append(
                train_one_model(
                    df_ml=df_ml_7d,
                    feature_cols=feature_cols,
                    target_col="TARGET_QTY_7D_AFTER_LAST",
                    model_name_prefix="prod_growth_plant_last7d_qty"
                )
            )

        if len(df_ml_total) > 0:
            results.append(
                train_one_model(
                    df_ml=df_ml_total,
                    feature_cols=feature_cols,
                    target_col="TARGET_QTY_TOTAL_CULT",
                    model_name_prefix="prod_growth_plant_totalcult_qty"
                )
            )

        print("\n8. 최종 결과")
        if len(results) > 0:
            print(pd.DataFrame(results))
        else:
            print("학습 가능한 데이터가 없습니다.")


if __name__ == "__main__":
    run_model()