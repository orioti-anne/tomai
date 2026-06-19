"""
Yield Model V2
- Target: yield_per_m2 (kg/m²) instead of total quantity
  → inference output * planting_area = total kg
- Removed: duplicate days_from_planting (== snapshot_day)
- Removed: leakage features (quantity_per_area, sales_per_area, avg_price_per_kg, etc.)
- Outlier filter: IQR × 3 on yield_per_m2
- Saves as: v2_yield_no_env_growth.joblib
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

current_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
app_root = os.path.join(package_root, "smartfarm")

if package_root not in sys.path:
    sys.path.append(package_root)

from smartfarm import create_app
from smartfarm.models import db

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

SNAPSHOT_DAYS = [14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98]


def season_from_month(month):
    return {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall", 10: "fall", 11: "fall"
    }.get(month)


def load_base_data():
    print("1. DB에서 base 데이터 로드")
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
            ps.cult_total_quantity
        FROM cultivations c
        JOIN farms f ON c.farm_id = f.farm_id
        JOIN prod_summary ps ON c.cult_id = ps.cult_id
        LEFT JOIN products p ON c.cult_id = p.cult_id
        WHERE ps.cult_total_quantity IS NOT NULL
          AND c.planting_date IS NOT NULL
          AND c.planting_area IS NOT NULL
        GROUP BY
            c.cult_id, c.farm_id, c.item, c.item_variety, c.crop_cycle,
            c.planting_date, c.planting_area, c.planting_density,
            c.house_type, c.house_form, c.survey_year,
            f.region_l1, f.region_l2, f.total_area, f.farm_num, f.first_survey_year,
            ps.cult_end_date, ps.cult_total_quantity
        ORDER BY c.cult_id
    """)

    df = pd.read_sql(query, db.engine)
    df.columns = [c.lower() for c in df.columns]

    for col in ["planting_date", "first_harvest_date", "cult_end_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df[df["cult_total_quantity"] > 0].copy()
    df = df[df["planting_area"] > 0].copy()

    # compute yield_per_m2 and remove outliers (IQR × 3)
    df["yield_per_m2"] = df["cult_total_quantity"] / df["planting_area"]
    q1 = df["yield_per_m2"].quantile(0.25)
    q3 = df["yield_per_m2"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    before = len(df)
    df = df[(df["yield_per_m2"] >= lower) & (df["yield_per_m2"] <= upper)].copy()
    print(f"   - 이상값 제거: {before - len(df)}건 (IQR×3 기준: [{lower:.1f}, {upper:.1f}] kg/m²)")
    print(f"   - base cult 수: {len(df):,}")
    return df


def make_snapshot_dataset(base_df):
    print("2. snapshot 데이터셋 생성")
    rows = []

    for _, base in base_df.iterrows():
        cult_id = base["cult_id"]
        planting_date = base["planting_date"]
        first_harvest_date = base["first_harvest_date"]
        cult_end_date = base["cult_end_date"]

        if pd.isna(planting_date):
            continue

        for snapshot_day in SNAPSHOT_DAYS:
            snapshot_date = planting_date + pd.Timedelta(days=int(snapshot_day))

            if pd.notna(first_harvest_date) and snapshot_date >= first_harvest_date:
                continue
            if pd.notna(cult_end_date) and snapshot_date > cult_end_date:
                continue

            row = {
                "cult_id": cult_id,
                "snapshot_day": snapshot_day,

                # target: yield per m²
                "target_yield_per_m2": float(base["yield_per_m2"]),

                # 재배 기본
                "item": base["item"],
                "item_variety": base["item_variety"],
                "crop_cycle": base["crop_cycle"],
                "planting_area": base["planting_area"],
                "planting_density": base["planting_density"],
                "house_type": base["house_type"],
                "house_form": base["house_form"],
                "survey_year": base["survey_year"],

                # 농가
                "region_l1": base["region_l1"],
                "region_l2": base["region_l2"],
                "total_area": base["total_area"],
                "farm_num": base["farm_num"],
                "first_survey_year": base["first_survey_year"],

                # 날짜 파생
                "planting_month": planting_date.month,
                "planting_season": season_from_month(planting_date.month),

                # 면적 비율
                "area_ratio": (
                    base["planting_area"] / base["total_area"]
                    if pd.notna(base["total_area"]) and base["total_area"] > 0 else np.nan
                ),
            }
            rows.append(row)

    snapshot_df = pd.DataFrame(rows)
    if len(snapshot_df) == 0:
        raise ValueError("snapshot 데이터가 0건입니다.")

    print(f"   - snapshot rows: {len(snapshot_df):,}")
    print(f"   - unique cult_id: {snapshot_df['cult_id'].nunique():,}")
    return snapshot_df


FEATURE_COLS = [
    "crop_cycle", "planting_area", "planting_density",
    "survey_year", "total_area", "farm_num", "first_survey_year",
    "planting_month", "snapshot_day", "area_ratio",
    "item", "item_variety", "house_type", "house_form",
    "region_l1", "region_l2", "planting_season",
]

CATEGORICAL_FEATURES = [
    "item", "item_variety", "house_type", "house_form",
    "region_l1", "region_l2", "planting_season",
]

NUMERIC_FEATURES = [c for c in FEATURE_COLS if c not in CATEGORICAL_FEATURES]


def train_model(snapshot_df):
    print("3. 모델 학습 (타겟: yield_per_m²)")

    X = snapshot_df[FEATURE_COLS].copy()
    y_raw = snapshot_df["target_yield_per_m2"].astype(float)
    y = np.log1p(y_raw)
    groups = snapshot_df["cult_id"]

    # 전부 null인 컬럼 제거
    all_null_cols = [c for c in X.columns if X[c].notna().sum() == 0]
    if all_null_cols:
        print(f"   - null 컬럼 제거: {all_null_cols}")

    num_features = [c for c in NUMERIC_FEATURES if c not in all_null_cols]
    cat_features = [c for c in CATEGORICAL_FEATURES if c not in all_null_cols]
    feat_cols = num_features + cat_features
    X = X[feat_cols].copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test_log = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    print(f"   - train: {len(X_train):,} rows / {train_groups.nunique():,} cults")
    print(f"   - test : {len(X_test):,} rows / {test_groups.nunique():,} cults")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_features),
        ]
    )

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    pred_log = pipeline.predict(X_test)
    y_test = np.expm1(y_test_log)
    pred = np.maximum(np.expm1(pred_log), 0)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    print("=" * 60)
    print("생산량 예측 모델 V2 결과 (타겟: yield_per_m²)")
    print(f"R2  : {r2:.4f}")
    print(f"MAE : {mae:.4f} kg/m²")
    print("=" * 60)

    result_df = X_test.copy()
    result_df["cult_id"] = test_groups.values
    result_df["actual"] = y_test.values
    result_df["pred"] = pred
    result_df["abs_error"] = np.abs(result_df["actual"] - result_df["pred"])

    print("\n오차 큰 상위 10건")
    print(result_df.sort_values("abs_error", ascending=False)[[
        "cult_id", "snapshot_day", "planting_area", "actual", "pred", "abs_error"
    ]].head(10))

    # feature importance
    ohe_names = list(
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(cat_features)
    )
    all_feature_names = num_features + ohe_names
    importances = pipeline.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    print("\n상위 중요 feature (V2)")
    print(imp_df.head(20).to_string(index=False))

    return pipeline, result_df, r2, mae, feat_cols


def save_artifacts(pipeline, result_df, r2, mae):
    print("4. 저장")
    model_dir = os.path.join(app_root, "ml", "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "v2_yield_no_env_growth.joblib")
    image_path = os.path.join(model_dir, "v2_yield_result.png")
    csv_path = os.path.join(model_dir, "v2_yield_result.csv")

    joblib.dump(pipeline, model_path)
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 8))
    plt.scatter(result_df["actual"], result_df["pred"], alpha=0.6)
    min_v = min(float(result_df["actual"].min()), float(result_df["pred"].min()))
    max_v = max(float(result_df["actual"].max()), float(result_df["pred"].max()))
    plt.plot([min_v, max_v], [min_v, max_v], "--", linewidth=2)
    plt.xlabel("실제 yield/m² (kg/m²)")
    plt.ylabel("예측 yield/m² (kg/m²)")
    plt.title(f"생산량 예측 V2 (per m²)\nR2={r2:.4f} / MAE={mae:.4f} kg/m²")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close()

    print(f"모델 저장: {model_path}")
    print(f"이미지  : {image_path}")
    print(f"CSV     : {csv_path}")


def main():
    app = create_app(enable_scheduler=False)
    with app.app_context():
        base_df = load_base_data()
        snapshot_df = make_snapshot_dataset(base_df)
        pipeline, result_df, r2, mae, feat_cols = train_model(snapshot_df)
        save_artifacts(pipeline, result_df, r2, mae)


if __name__ == "__main__":
    main()
