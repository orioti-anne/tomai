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

    print(f"   - base cult 수: {len(df):,}")
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


# --------------------------------------------------
# 5. snapshot 데이터셋 생성
#    - 환경/생육 없이 재배 + 생산기준만 사용
# --------------------------------------------------
def make_snapshot_dataset(base_df):
    print("2. cult_id × 날짜(snapshot) 학습 데이터 생성 (환경/생육 제외)")

    rows = []

    for _, base in base_df.iterrows():
        cult_id = base["cult_id"]
        planting_date = base["planting_date"]
        first_harvest_date = base["first_harvest_date"]
        cult_end_date = base["cult_end_date"]

        if pd.isna(planting_date):
            continue

        total_days_to_end = np.nan
        if pd.notna(cult_end_date):
            total_days_to_end = (cult_end_date - planting_date).days

        days_to_first_harvest = np.nan
        if pd.notna(first_harvest_date):
            days_to_first_harvest = (first_harvest_date - planting_date).days

        quantity_per_area = np.nan
        sales_per_area = np.nan
        avg_price_per_kg = np.nan

        if pd.notna(base["planting_area"]) and base["planting_area"] > 0:
            quantity_per_area = base["cult_total_quantity"] / base["planting_area"]
            if pd.notna(base["cult_total_sales"]):
                sales_per_area = base["cult_total_sales"] / base["planting_area"]

        if pd.notna(base["cult_total_quantity"]) and base["cult_total_quantity"] > 0 and pd.notna(base["cult_total_sales"]):
            avg_price_per_kg = base["cult_total_sales"] / base["cult_total_quantity"]

        for snapshot_day in SNAPSHOT_DAYS:
            snapshot_date = planting_date + pd.Timedelta(days=int(snapshot_day))

            if pd.notna(first_harvest_date) and snapshot_date >= first_harvest_date:
                continue

            if pd.notna(cult_end_date) and snapshot_date > cult_end_date:
                continue

            row = {
                "cult_id": cult_id,
                "snapshot_date": snapshot_date,
                "snapshot_day": snapshot_day,

                # target
                "target_quantity": float(base["cult_total_quantity"]),

                # 재배 기본 정보
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

                # 날짜 파생
                "planting_month": planting_date.month,
                "planting_season": season_from_month(planting_date.month),
                "days_from_planting": snapshot_day,

                # 면적/규모 파생
                "area_ratio": (
                    base["planting_area"] / base["total_area"]
                    if pd.notna(base["total_area"]) and base["total_area"] > 0 else np.nan
                ),

                # 생산 요약 파생
                "cult_total_sales": base["cult_total_sales"],
                "quantity_per_area": quantity_per_area,
                "sales_per_area": sales_per_area,
                "avg_price_per_kg": avg_price_per_kg,
                "days_to_first_harvest": days_to_first_harvest,
                "total_days_to_end": total_days_to_end,
            }

            rows.append(row)

    snapshot_df = pd.DataFrame(rows)

    if len(snapshot_df) == 0:
        raise ValueError("snapshot 데이터가 0건입니다. base 데이터 상태를 확인해줘.")

    print(f"   - snapshot rows: {len(snapshot_df):,}")
    print(f"   - unique cult_id: {snapshot_df['cult_id'].nunique():,}")

    return snapshot_df


# --------------------------------------------------
# 6. 모델 학습
#    - 순수 재배 정보만 사용
#    - 환경 / 생육 / 생산결과 파생값 전부 제거
# --------------------------------------------------
def train_model(snapshot_df):
    print("3. 모델 학습 (순수 재배정보만 사용)")

    feature_cols = [
        # 재배 기본 정보
        "crop_cycle",
        "planting_area",
        "planting_density",
        "survey_year",
        "total_area",
        "farm_num",
        "first_survey_year",
        "planting_month",
        "snapshot_day",
        "days_from_planting",
        "area_ratio",

        # 범주형
        "item",
        "item_variety",
        "house_type",
        "house_form",
        "region_l1",
        "region_l2",
        "planting_season",
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

    numeric_features = [
        c for c in feature_cols
        if c not in categorical_features
    ]

    X = snapshot_df[feature_cols].copy()
    y_raw = snapshot_df["target_quantity"].astype(float)
    y = np.log1p(y_raw)
    groups = snapshot_df["cult_id"]

    # 전부 null인 컬럼 제거
    all_null_cols = [c for c in X.columns if X[c].notna().sum() == 0]
    if all_null_cols:
        print(f"   - 제거되는 null 컬럼: {all_null_cols}")

        X = X.drop(columns=all_null_cols)

        feature_cols = [c for c in feature_cols if c not in all_null_cols]
        categorical_features = [c for c in categorical_features if c not in all_null_cols]
        numeric_features = [c for c in numeric_features if c not in all_null_cols]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test_log = y.iloc[test_idx].copy()

    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    print(f"   - train rows : {len(X_train):,}")
    print(f"   - test rows  : {len(X_test):,}")
    print(f"   - train cult : {train_groups.nunique():,}")
    print(f"   - test cult  : {test_groups.nunique():,}")

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

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # 큰 생산량 건에 약간 가중치
    sample_weight = np.sqrt(
        snapshot_df.iloc[train_idx]["target_quantity"].astype(float).values
    )

    pipeline.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weight
    )

    pred_log = pipeline.predict(X_test)

    y_test = np.expm1(y_test_log)
    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    print("=" * 60)
    print("생산량 예측 모델 결과 (순수 재배정보만)")
    print(f"R2  : {r2:.4f}")
    print(f"MAE : {mae:,.2f} kg")
    print("=" * 60)

    result_df = X_test.copy()
    result_df["cult_id"] = test_groups.values
    result_df["actual"] = y_test.values
    result_df["pred"] = pred
    result_df["abs_error"] = np.abs(result_df["actual"] - result_df["pred"])

    print("\n오차 큰 상위 10건")
    print(
        result_df.sort_values("abs_error", ascending=False)[[
            "cult_id",
            "snapshot_day",
            "planting_area",
            "item_variety",
            "region_l1",
            "actual",
            "pred",
            "abs_error"
        ]].head(10)
    )

    # feature importance 출력
    feature_names = (
        numeric_features +
        list(
            pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(categorical_features)
        )
    )

    importances = pipeline.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n상위 중요 feature")
    print(importance_df.head(20))

    return pipeline, result_df, r2, mae

# --------------------------------------------------
# 7. 결과 저장
# --------------------------------------------------
def save_artifacts(pipeline, result_df, r2, mae):
    print("4. 모델/이미지 저장")

    model_dir = os.path.join(app_root, "ml", "models")
    os.makedirs(model_dir, exist_ok=True)

    print(f"[MODEL_DIR] {model_dir}")

    model_path = os.path.join(model_dir, "yield_prediction_model_no_env_growth.joblib")
    image_path = os.path.join(model_dir, "yield_prediction_result_no_env_growth.png")
    csv_path = os.path.join(model_dir, "yield_prediction_result_no_env_growth.csv")

    joblib.dump(pipeline, model_path)
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(12, 9))
    plt.scatter(result_df["actual"], result_df["pred"], alpha=0.6)

    min_val = min(float(result_df["actual"].min()), float(result_df["pred"].min()))
    max_val = max(float(result_df["actual"].max()), float(result_df["pred"].max()))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=2
    )

    plt.xlabel("실제 총 생산량 (kg)")
    plt.ylabel("예측 총 생산량 (kg)")
    plt.title(f"생산량 예측 결과 (환경/생육 제외)\nR2 = {r2:.4f} / MAE = {mae:,.1f} kg")
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
        snapshot_df = make_snapshot_dataset(base_df)
        pipeline, result_df, r2, mae = train_model(snapshot_df)
        save_artifacts(pipeline, result_df, r2, mae)


if __name__ == "__main__":
    main()