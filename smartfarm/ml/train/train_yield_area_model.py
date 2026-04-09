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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit


# --------------------------------------------------
# 1. 경로 / 환경 설정
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))                   # .../smartfarm/ml/train
package_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # .../AI_project1
app_root = os.path.join(package_root, "smartfarm")                        # .../AI_project1/smartfarm

if package_root not in sys.path:
    sys.path.append(package_root)

from smartfarm import create_app
from smartfarm.models import db

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------
# 2. 설정값
# --------------------------------------------------
SNAPSHOT_DAYS = [14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84]
MIN_ENV_DAYS = 7


# --------------------------------------------------
# 3. DB 로드
# --------------------------------------------------
def load_base_data():
    print("1. 오라클 DB에서 base 데이터 로드")

    query = text("""
        SELECT
            c.cult_id,
            c.farm_id,
            c.item,
            c.item_variety,
            c.crop_cycle,
            c.planting_date,
            c.planting_area,
            c.planting_density,
            c.house_type,
            c.house_form,
            c.survey_year,

            f.region_l1,
            f.region_l2,
            f.total_area,
            f.farm_num,
            f.first_survey_year,

            MIN(p.production_date) AS first_harvest_date,
            ps.cult_end_date,
            ps.cult_total_quantity,
            ps.cult_total_sales
        FROM cultivations c
        JOIN farms f
            ON c.farm_id = f.farm_id
        JOIN prod_summary ps
            ON c.cult_id = ps.cult_id
        LEFT JOIN products p
            ON c.cult_id = p.cult_id
        WHERE ps.cult_total_quantity IS NOT NULL
          AND c.planting_date IS NOT NULL
          AND c.planting_area IS NOT NULL
        GROUP BY
            c.cult_id,
            c.farm_id,
            c.item,
            c.item_variety,
            c.crop_cycle,
            c.planting_date,
            c.planting_area,
            c.planting_density,
            c.house_type,
            c.house_form,
            c.survey_year,
            f.region_l1,
            f.region_l2,
            f.total_area,
            f.farm_num,
            f.first_survey_year,
            ps.cult_end_date,
            ps.cult_total_quantity,
            ps.cult_total_sales
        ORDER BY c.cult_id
    """)

    df = pd.read_sql(query, db.engine)
    df.columns = [c.lower() for c in df.columns]

    for col in ["planting_date", "first_harvest_date", "cult_end_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df[df["cult_total_quantity"] > 0].copy()
    df = df[df["planting_area"] > 0].copy()

    # 면적당 생산량 타깃
    df["target_yield_per_area"] = df["cult_total_quantity"] / df["planting_area"]

    print(f"   - base cult 수: {len(df):,}")
    return df


def load_env_data():
    print("2. 오라클 DB에서 ENV_SUMMARY 로드")

    query = text("""
        SELECT
            cult_id,
            measure_date,
            daily_out_temp,
            daily_acc_solar,
            daily_rain_detection,
            daily_in_temp,
            daily_in_humidity,
            daily_in_co2,
            daily_soil_temp,
            acc_temp,
            acc_solar
        FROM env_summary
        ORDER BY cult_id, measure_date
    """)

    df = pd.read_sql(query, db.engine)
    df.columns = [c.lower() for c in df.columns]
    df["measure_date"] = pd.to_datetime(df["measure_date"], errors="coerce")

    print(f"   - env_summary rows: {len(df):,}")
    return df


def load_growth_data():
    print("3. 오라클 DB에서 GROW_SUMMARY 로드")

    query = text("""
        SELECT
            cult_id,
            inspect_date,
            growth_days,
            plant_num,
            branch_num,
            plant_height,
            growth_length,
            leaf_count,
            leaf_length,
            leaf_width,
            branch_width,
            cluster_height,
            cluster_num,
            flowers_per_cluster,
            blooming_per_cluster,
            fruits_per_cluster
        FROM grow_summary
        ORDER BY cult_id, inspect_date
    """)

    df = pd.read_sql(query, db.engine)
    df.columns = [c.lower() for c in df.columns]
    df["inspect_date"] = pd.to_datetime(df["inspect_date"], errors="coerce")

    print(f"   - grow_summary rows: {len(df):,}")
    return df


# --------------------------------------------------
# 4. 보조 함수
# --------------------------------------------------
def season_from_month(month):
    return {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall", 10: "fall", 11: "fall"
    }.get(month)


def get_recent_env_features(env_slice, snapshot_date):
    if len(env_slice) == 0:
        return {}

    recent_7 = env_slice[env_slice["measure_date"] > (snapshot_date - pd.Timedelta(days=7))]
    recent_14 = env_slice[env_slice["measure_date"] > (snapshot_date - pd.Timedelta(days=14))]

    return {
        "recent7_out_temp_mean": recent_7["daily_out_temp"].mean() if len(recent_7) else np.nan,
        "recent7_in_temp_mean": recent_7["daily_in_temp"].mean() if len(recent_7) else np.nan,
        "recent7_humidity_mean": recent_7["daily_in_humidity"].mean() if len(recent_7) else np.nan,
        "recent7_co2_mean": recent_7["daily_in_co2"].mean() if len(recent_7) else np.nan,
        "recent7_soil_temp_mean": recent_7["daily_soil_temp"].mean() if len(recent_7) else np.nan,
        "recent7_solar_mean": recent_7["daily_acc_solar"].mean() if len(recent_7) else np.nan,
        "recent7_rain_days": recent_7["daily_rain_detection"].fillna(0).sum() if len(recent_7) else 0,

        "recent14_out_temp_mean": recent_14["daily_out_temp"].mean() if len(recent_14) else np.nan,
        "recent14_in_temp_mean": recent_14["daily_in_temp"].mean() if len(recent_14) else np.nan,
        "recent14_humidity_mean": recent_14["daily_in_humidity"].mean() if len(recent_14) else np.nan,
        "recent14_co2_mean": recent_14["daily_in_co2"].mean() if len(recent_14) else np.nan,
        "recent14_soil_temp_mean": recent_14["daily_soil_temp"].mean() if len(recent_14) else np.nan,
        "recent14_solar_mean": recent_14["daily_acc_solar"].mean() if len(recent_14) else np.nan,
        "recent14_rain_days": recent_14["daily_rain_detection"].fillna(0).sum() if len(recent_14) else 0,
    }


def get_growth_features(cult_growth, snapshot_date):
    result = {
        "has_growth": 0,
        "growth_days_latest": np.nan,
        "plant_num_latest": np.nan,
        "branch_num_latest": np.nan,
        "plant_height_latest": np.nan,
        "growth_length_latest": np.nan,
        "leaf_count_latest": np.nan,
        "leaf_length_latest": np.nan,
        "leaf_width_latest": np.nan,
        "branch_width_latest": np.nan,
        "cluster_height_latest": np.nan,
        "cluster_num_latest": np.nan,
        "flowers_per_cluster_latest": np.nan,
        "blooming_per_cluster_latest": np.nan,
        "fruits_per_cluster_latest": np.nan,

        "plant_height_diff": np.nan,
        "growth_length_diff": np.nan,
        "leaf_count_diff": np.nan,
        "cluster_num_diff": np.nan,
        "fruits_per_cluster_diff": np.nan,
        "growth_measure_gap_days": np.nan,

        "plant_height_diff_per_day": np.nan,
        "leaf_count_diff_per_day": np.nan,
        "cluster_num_diff_per_day": np.nan,
    }

    if cult_growth is None:
        return result

    growth_slice = cult_growth[cult_growth["inspect_date"] <= snapshot_date].copy()
    if len(growth_slice) == 0:
        return result

    growth_slice = growth_slice.sort_values("inspect_date")
    latest = growth_slice.iloc[-1]

    result.update({
        "has_growth": 1,
        "growth_days_latest": latest.get("growth_days"),
        "plant_num_latest": latest.get("plant_num"),
        "branch_num_latest": latest.get("branch_num"),
        "plant_height_latest": latest.get("plant_height"),
        "growth_length_latest": latest.get("growth_length"),
        "leaf_count_latest": latest.get("leaf_count"),
        "leaf_length_latest": latest.get("leaf_length"),
        "leaf_width_latest": latest.get("leaf_width"),
        "branch_width_latest": latest.get("branch_width"),
        "cluster_height_latest": latest.get("cluster_height"),
        "cluster_num_latest": latest.get("cluster_num"),
        "flowers_per_cluster_latest": latest.get("flowers_per_cluster"),
        "blooming_per_cluster_latest": latest.get("blooming_per_cluster"),
        "fruits_per_cluster_latest": latest.get("fruits_per_cluster"),
    })

    if len(growth_slice) >= 2:
        prev = growth_slice.iloc[-2]
        days_gap = (latest["inspect_date"] - prev["inspect_date"]).days
        days_gap = days_gap if days_gap > 0 else np.nan

        plant_height_diff = (
            latest.get("plant_height") - prev.get("plant_height")
            if pd.notna(latest.get("plant_height")) and pd.notna(prev.get("plant_height"))
            else np.nan
        )
        growth_length_diff = (
            latest.get("growth_length") - prev.get("growth_length")
            if pd.notna(latest.get("growth_length")) and pd.notna(prev.get("growth_length"))
            else np.nan
        )
        leaf_count_diff = (
            latest.get("leaf_count") - prev.get("leaf_count")
            if pd.notna(latest.get("leaf_count")) and pd.notna(prev.get("leaf_count"))
            else np.nan
        )
        cluster_num_diff = (
            latest.get("cluster_num") - prev.get("cluster_num")
            if pd.notna(latest.get("cluster_num")) and pd.notna(prev.get("cluster_num"))
            else np.nan
        )
        fruits_per_cluster_diff = (
            latest.get("fruits_per_cluster") - prev.get("fruits_per_cluster")
            if pd.notna(latest.get("fruits_per_cluster")) and pd.notna(prev.get("fruits_per_cluster"))
            else np.nan
        )

        result.update({
            "plant_height_diff": plant_height_diff,
            "growth_length_diff": growth_length_diff,
            "leaf_count_diff": leaf_count_diff,
            "cluster_num_diff": cluster_num_diff,
            "fruits_per_cluster_diff": fruits_per_cluster_diff,
            "growth_measure_gap_days": days_gap,
        })

        if pd.notna(days_gap) and days_gap > 0:
            result.update({
                "plant_height_diff_per_day": plant_height_diff / days_gap if pd.notna(plant_height_diff) else np.nan,
                "leaf_count_diff_per_day": leaf_count_diff / days_gap if pd.notna(leaf_count_diff) else np.nan,
                "cluster_num_diff_per_day": cluster_num_diff / days_gap if pd.notna(cluster_num_diff) else np.nan,
            })

    return result


# --------------------------------------------------
# 5. snapshot 데이터셋 생성
# --------------------------------------------------
def make_snapshot_dataset(base_df, env_df, growth_df):
    print("4. cult_id × 날짜(snapshot) 학습 데이터 생성")

    rows = []

    env_group = {
        cult_id: g.sort_values("measure_date").copy()
        for cult_id, g in env_df.groupby("cult_id")
    }
    growth_group = {
        cult_id: g.sort_values("inspect_date").copy()
        for cult_id, g in growth_df.groupby("cult_id")
    }

    for _, base in base_df.iterrows():
        cult_id = base["cult_id"]
        planting_date = base["planting_date"]
        first_harvest_date = base["first_harvest_date"]
        cult_end_date = base["cult_end_date"]

        if pd.isna(planting_date):
            continue

        cult_env = env_group.get(cult_id)
        cult_growth = growth_group.get(cult_id)

        for snapshot_day in SNAPSHOT_DAYS:
            snapshot_date = planting_date + pd.Timedelta(days=int(snapshot_day))

            if pd.notna(first_harvest_date) and snapshot_date >= first_harvest_date:
                continue

            if pd.notna(cult_end_date) and snapshot_date > cult_end_date:
                continue

            if cult_env is None:
                continue

            env_slice = cult_env[
                (cult_env["measure_date"] >= planting_date) &
                (cult_env["measure_date"] <= snapshot_date)
            ].copy()

            if len(env_slice) < MIN_ENV_DAYS:
                continue

            latest_env = env_slice.sort_values("measure_date").iloc[-1]
            recent_env = get_recent_env_features(env_slice, snapshot_date)
            growth_feat = get_growth_features(cult_growth, snapshot_date)

            row = {
                "cult_id": cult_id,
                "snapshot_date": snapshot_date,
                "snapshot_day": snapshot_day,

                # 타깃
                "target_yield_per_area": float(base["target_yield_per_area"]),
                "actual_total_quantity": float(base["cult_total_quantity"]),

                # 기본 정보
                "item": base["item"],
                "item_variety": base["item_variety"],
                "crop_cycle": base["crop_cycle"],
                "planting_area": base["planting_area"],
                "planting_density": base["planting_density"],
                "house_type": base["house_type"],
                "house_form": base["house_form"],
                "survey_year": base["survey_year"],

                # 농가 정보
                "region_l1": base["region_l1"],
                "region_l2": base["region_l2"],
                "total_area": base["total_area"],
                "farm_num": base["farm_num"],
                "first_survey_year": base["first_survey_year"],

                # 날짜 정보
                "planting_month": planting_date.month,
                "planting_season": season_from_month(planting_date.month),

                # 면적 관련
                "area_ratio": (
                    base["planting_area"] / base["total_area"]
                    if pd.notna(base["total_area"]) and base["total_area"] > 0 else np.nan
                ),

                # 환경 전체 집계
                "env_days": len(env_slice),
                "acc_temp_to_snapshot": latest_env.get("acc_temp"),
                "acc_solar_to_snapshot": latest_env.get("acc_solar"),
                "avg_daily_out_temp": env_slice["daily_out_temp"].mean(),
                "avg_daily_in_temp": env_slice["daily_in_temp"].mean(),
                "avg_daily_in_humidity": env_slice["daily_in_humidity"].mean(),
                "avg_daily_in_co2": env_slice["daily_in_co2"].mean(),
                "avg_daily_soil_temp": env_slice["daily_soil_temp"].mean(),
                "avg_daily_acc_solar": env_slice["daily_acc_solar"].mean(),
                "rain_days_total": env_slice["daily_rain_detection"].fillna(0).sum(),
            }

            row.update(recent_env)
            row.update(growth_feat)
            rows.append(row)

    snapshot_df = pd.DataFrame(rows)

    if len(snapshot_df) == 0:
        raise ValueError("snapshot 데이터가 0건입니다. ENV_SUMMARY / GROW_SUMMARY 적재 상태를 확인해줘.")

    print(f"   - snapshot rows: {len(snapshot_df):,}")
    print(f"   - unique cult_id: {snapshot_df['cult_id'].nunique():,}")
    return snapshot_df


# --------------------------------------------------
# 6. 모델 학습
# --------------------------------------------------
def train_model(snapshot_df):
    print("5. 모델 학습")

    feature_cols = [
        # 기본
        "crop_cycle",
        "planting_area",
        "planting_density",
        "survey_year",
        "total_area",
        "farm_num",
        "first_survey_year",
        "planting_month",
        "snapshot_day",
        "area_ratio",

        # 환경 전체
        "env_days",
        "acc_temp_to_snapshot",
        "acc_solar_to_snapshot",
        "avg_daily_out_temp",
        "avg_daily_in_temp",
        "avg_daily_in_humidity",
        "avg_daily_in_co2",
        "avg_daily_soil_temp",
        "avg_daily_acc_solar",
        "rain_days_total",

        # 환경 최근
        "recent7_out_temp_mean",
        "recent7_in_temp_mean",
        "recent7_humidity_mean",
        "recent7_co2_mean",
        "recent7_soil_temp_mean",
        "recent7_solar_mean",
        "recent7_rain_days",

        "recent14_out_temp_mean",
        "recent14_in_temp_mean",
        "recent14_humidity_mean",
        "recent14_co2_mean",
        "recent14_soil_temp_mean",
        "recent14_solar_mean",
        "recent14_rain_days",

        # 생육 최신값
        "has_growth",
        "growth_days_latest",
        "plant_num_latest",
        "branch_num_latest",
        "plant_height_latest",
        "growth_length_latest",
        "leaf_count_latest",
        "leaf_length_latest",
        "leaf_width_latest",
        "branch_width_latest",
        "cluster_height_latest",
        "cluster_num_latest",
        "flowers_per_cluster_latest",
        "blooming_per_cluster_latest",
        "fruits_per_cluster_latest",

        # 생육 변화량
        "plant_height_diff",
        "growth_length_diff",
        "leaf_count_diff",
        "cluster_num_diff",
        "fruits_per_cluster_diff",
        "growth_measure_gap_days",
        "plant_height_diff_per_day",
        "leaf_count_diff_per_day",
        "cluster_num_diff_per_day",

        # 범주형
        "item",
        "item_variety",
        "house_type",
        "house_form",
        "region_l1",
        "region_l2",
        "planting_season",
    ]

    numeric_features = [
        c for c in feature_cols
        if c not in ["item", "item_variety", "house_type", "house_form", "region_l1", "region_l2", "planting_season"]
    ]

    categorical_features = [
        "item",
        "item_variety",
        "house_type",
        "house_form",
        "region_l1",
        "region_l2",
        "planting_season",
    ]

    X = snapshot_df[feature_cols].copy()
    y_raw = snapshot_df["target_yield_per_area"].astype(float)
    y = np.log1p(y_raw)
    groups = snapshot_df["cult_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test_log = y.iloc[test_idx].copy()

    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    print(f"   - train rows: {len(X_train):,}")
    print(f"   - test rows : {len(X_test):,}")
    print(f"   - train cult: {train_groups.nunique():,}")
    print(f"   - test cult : {test_groups.nunique():,}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numeric_features
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_features
            )
        ]
    )

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # 큰 농가의 중요도를 약간 높임
    sample_weight = np.sqrt(snapshot_df.iloc[train_idx]["planting_area"].astype(float).values)

    pipeline.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weight
    )

    pred_log = pipeline.predict(X_test)

    # 면적당 예측값 복원
    y_test_area = np.expm1(y_test_log)
    pred_area = np.expm1(pred_log)
    pred_area = np.maximum(pred_area, 0)

    # 총 생산량으로 환산
    planting_area_test = snapshot_df.iloc[test_idx]["planting_area"].astype(float).values
    actual_total = snapshot_df.iloc[test_idx]["actual_total_quantity"].astype(float).values
    pred_total = pred_area * planting_area_test

    r2 = r2_score(actual_total, pred_total)
    mae = mean_absolute_error(actual_total, pred_total)

    print("=" * 60)
    print("생산량 예측 모델 결과 (면적당 생산량 → 총 생산량 환산)")
    print(f"R2  : {r2:.4f}")
    print(f"MAE : {mae:,.2f} kg")
    print("=" * 60)

    result_df = X_test.copy()
    result_df["cult_id"] = test_groups.values
    result_df["actual_yield_per_area"] = y_test_area.values
    result_df["pred_yield_per_area"] = pred_area
    result_df["actual_total"] = actual_total
    result_df["pred_total"] = pred_total
    result_df["abs_error"] = np.abs(result_df["actual_total"] - result_df["pred_total"])

    print("\n오차 큰 상위 10건")
    print(
        result_df.sort_values("abs_error", ascending=False)[
            [
                "cult_id", "snapshot_day", "planting_area",
                "item_variety", "region_l1",
                "actual_total", "pred_total", "abs_error"
            ]
        ].head(10)
    )

    return pipeline, result_df, r2, mae


# --------------------------------------------------
# 7. 결과 저장
# --------------------------------------------------
def save_artifacts(pipeline, result_df, r2, mae):
    print("6. 모델/이미지 저장")

    model_dir = os.path.join(app_root, "ml", "models")
    os.makedirs(model_dir, exist_ok=True)

    print(f"[MODEL_DIR] {model_dir}")

    model_path = os.path.join(model_dir, "yield_area_prediction_model.joblib")
    image_path = os.path.join(model_dir, "yield_area_prediction_result.png")
    csv_path = os.path.join(model_dir, "yield_area_prediction_result.csv")

    joblib.dump(pipeline, model_path)
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(12, 9))
    plt.scatter(result_df["actual_total"], result_df["pred_total"], alpha=0.6)

    min_val = min(float(result_df["actual_total"].min()), float(result_df["pred_total"].min()))
    max_val = max(float(result_df["actual_total"].max()), float(result_df["pred_total"].max()))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=2
    )

    plt.xlabel("실제 총 생산량 (kg)")
    plt.ylabel("예측 총 생산량 (kg)")
    plt.title(f"생산량 예측 결과 (면적당 생산량 기반)\nR2 = {r2:.4f} / MAE = {mae:,.1f} kg")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close()

    print(f"모델 저장 완료 : {model_path}")
    print(f"이미지 저장 완료: {image_path}")
    print(f"CSV 저장 완료   : {csv_path}")


# --------------------------------------------------
# 8. 실행
# --------------------------------------------------
def main():
    app = create_app(enable_scheduler=False)

    with app.app_context():
        base_df = load_base_data()
        env_df = load_env_data()
        growth_df = load_growth_data()

        snapshot_df = make_snapshot_dataset(base_df, env_df, growth_df)
        pipeline, result_df, r2, mae = train_model(snapshot_df)
        save_artifacts(pipeline, result_df, r2, mae)


if __name__ == "__main__":
    main()