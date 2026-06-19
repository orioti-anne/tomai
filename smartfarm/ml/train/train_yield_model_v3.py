"""
Yield Model V3
- Target: yield_per_area (kg/평) = cult_total_quantity / planting_area
- Features: 환경(env_summary) + 생육(grow_summary) + 재배기본정보
- Snapshot: 정식 후 14~98일 시점별 snapshot row 생성
- Outlier: IQR×3 제거
- Split: GroupShuffleSplit (cult 단위, 누수 방지)
- Saves: v3_yield_pipeline.joblib
"""
import os, sys, joblib
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

current_dir   = os.path.dirname(os.path.abspath(__file__))
package_root  = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if package_root not in sys.path:
    sys.path.append(package_root)

from smartfarm import create_app
from smartfarm.models import db

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

SNAPSHOT_DAYS = [14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98]


def season_from_month(m):
    return {12:"winter",1:"winter",2:"winter",
            3:"spring", 4:"spring", 5:"spring",
            6:"summer", 7:"summer", 8:"summer",
            9:"fall",  10:"fall",  11:"fall"}.get(m)


# ── 1. 기본 cult 정보 ────────────────────────────────────────
def load_base():
    q = text("""
        SELECT c.cult_id, c.farm_id, c.planting_date, c.planting_area,
               c.planting_density, c.house_type, c.house_form,
               c.crop_cycle, c.item, c.item_variety, c.survey_year,
               f.region_l1, f.region_l2, f.total_area,
               MIN(p.production_date) AS first_harvest_date,
               ps.cult_end_date, ps.cult_total_quantity
        FROM cultivations c
        JOIN farms f          ON c.farm_id = f.farm_id
        JOIN prod_summary ps  ON c.cult_id = ps.cult_id
        LEFT JOIN products p  ON c.cult_id = p.cult_id
        WHERE ps.cult_total_quantity > 0
          AND c.planting_date IS NOT NULL
          AND c.planting_area > 0
          AND EXISTS (SELECT 1 FROM env_summary e WHERE e.cult_id = c.cult_id)
        GROUP BY c.cult_id, c.farm_id, c.planting_date, c.planting_area,
                 c.planting_density, c.house_type, c.house_form,
                 c.crop_cycle, c.item, c.item_variety, c.survey_year,
                 f.region_l1, f.region_l2, f.total_area,
                 ps.cult_end_date, ps.cult_total_quantity
        ORDER BY c.cult_id
    """)
    df = pd.read_sql(q, db.engine)
    df.columns = [c.lower() for c in df.columns]
    for col in ["planting_date","first_harvest_date","cult_end_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # yield_per_area 계산 + IQR×3 이상치 제거
    df["yield_per_area"] = df["cult_total_quantity"] / df["planting_area"]
    q1, q3 = df["yield_per_area"].quantile([0.25, 0.75])
    iqr = q3 - q1
    before = len(df)
    df = df[(df["yield_per_area"] >= q1 - 3*iqr) & (df["yield_per_area"] <= q3 + 3*iqr)].copy()
    print(f"   기본 cult: {before} → {len(df)} (IQR×3 필터, 범위 [{q1-3*iqr:.1f}, {q3+3*iqr:.1f}] kg/평)")
    return df.reset_index(drop=True)


# ── 2. 환경 데이터 전체 로드 ──────────────────────────────────
def load_env_all(cult_ids):
    ids = ",".join(str(i) for i in cult_ids)
    q = text(f"""
        SELECT cult_id, measure_date,
               daily_in_temp, daily_in_humidity, daily_in_co2,
               daily_acc_solar, daily_rain_detection,
               acc_temp, acc_solar
        FROM env_summary
        WHERE cult_id IN ({ids})
        ORDER BY cult_id, measure_date
    """)
    df = pd.read_sql(q, db.engine)
    df.columns = [c.lower() for c in df.columns]
    df["measure_date"] = pd.to_datetime(df["measure_date"])
    return df


# ── 3. 생육 데이터 전체 로드 (inspect_date별 식물 평균) ─────────
def load_grow_all(cult_ids):
    ids = ",".join(str(i) for i in cult_ids)
    q = text(f"""
        SELECT cult_id, inspect_date,
               AVG(plant_height)          AS plant_height,
               AVG(growth_length)         AS growth_length,
               AVG(leaf_count)            AS leaf_count,
               AVG(cluster_num)           AS cluster_num,
               AVG(fruits_per_cluster)    AS fruits_per_cluster,
               AVG(branch_width)          AS branch_width
        FROM grow_summary
        WHERE cult_id IN ({ids})
        GROUP BY cult_id, inspect_date
        ORDER BY cult_id, inspect_date
    """)
    df = pd.read_sql(q, db.engine)
    df.columns = [c.lower() for c in df.columns]
    df["inspect_date"] = pd.to_datetime(df["inspect_date"])
    return df


# ── 4. 환경 피처 계산 (snapshot_date까지) ───────────────────────
def env_features(env_cult: pd.DataFrame, snapshot_date: pd.Timestamp) -> dict:
    df = env_cult[env_cult["measure_date"] <= snapshot_date].copy()
    if df.empty:
        return {}

    r7  = df[df["measure_date"] > snapshot_date - pd.Timedelta(days=7)]
    r14 = df[df["measure_date"] > snapshot_date - pd.Timedelta(days=14)]
    latest = df.iloc[-1]

    def safe_mean(s): return float(s.mean()) if len(s) > 0 else np.nan

    return {
        "env_days":            len(df),
        "acc_temp":            float(latest["acc_temp"])       if pd.notna(latest["acc_temp"])  else np.nan,
        "acc_solar":           float(latest["acc_solar"])      if pd.notna(latest["acc_solar"]) else np.nan,
        "avg_in_temp":         safe_mean(pd.to_numeric(df["daily_in_temp"],   errors="coerce")),
        "avg_in_humidity":     safe_mean(pd.to_numeric(df["daily_in_humidity"],errors="coerce")),
        "avg_in_co2":          safe_mean(pd.to_numeric(df["daily_in_co2"],    errors="coerce")),
        "avg_acc_solar":       safe_mean(pd.to_numeric(df["daily_acc_solar"], errors="coerce")),
        "rain_days":           float(pd.to_numeric(df["daily_rain_detection"],errors="coerce").fillna(0).sum()),
        "r7_in_temp":          safe_mean(pd.to_numeric(r7["daily_in_temp"],   errors="coerce")),
        "r7_solar":            safe_mean(pd.to_numeric(r7["daily_acc_solar"], errors="coerce")),
        "r14_in_temp":         safe_mean(pd.to_numeric(r14["daily_in_temp"],  errors="coerce")),
        "r14_solar":           safe_mean(pd.to_numeric(r14["daily_acc_solar"],errors="coerce")),
    }


# ── 5. 생육 피처 계산 (snapshot_date까지 최신) ──────────────────
def grow_features(grow_cult: pd.DataFrame, snapshot_date: pd.Timestamp) -> dict:
    null_row = {
        "has_growth": 0,
        "plant_height":         np.nan, "growth_length":      np.nan,
        "leaf_count":           np.nan, "cluster_num":        np.nan,
        "fruits_per_cluster":   np.nan, "branch_width":       np.nan,
        "plant_height_diff":    np.nan, "growth_length_diff": np.nan,
        "cluster_num_diff":     np.nan, "fruits_per_cluster_diff": np.nan,
        "grow_obs_days":        np.nan,
    }
    df = grow_cult[grow_cult["inspect_date"] <= snapshot_date].copy()
    if df.empty:
        return null_row

    df = df.sort_values("inspect_date")
    latest = df.iloc[-1]

    feats = {
        "has_growth":            1,
        "plant_height":          float(latest["plant_height"])       if pd.notna(latest["plant_height"])       else np.nan,
        "growth_length":         float(latest["growth_length"])      if pd.notna(latest["growth_length"])      else np.nan,
        "leaf_count":            float(latest["leaf_count"])         if pd.notna(latest["leaf_count"])         else np.nan,
        "cluster_num":           float(latest["cluster_num"])        if pd.notna(latest["cluster_num"])        else np.nan,
        "fruits_per_cluster":    float(latest["fruits_per_cluster"]) if pd.notna(latest["fruits_per_cluster"]) else np.nan,
        "branch_width":          float(latest["branch_width"])       if pd.notna(latest["branch_width"])       else np.nan,
        "plant_height_diff":     np.nan,
        "growth_length_diff":    np.nan,
        "cluster_num_diff":      np.nan,
        "fruits_per_cluster_diff": np.nan,
        "grow_obs_days":         float((df["inspect_date"].iloc[-1] - df["inspect_date"].iloc[0]).days) if len(df) > 1 else 0.0,
    }

    if len(df) >= 2:
        prev = df.iloc[-2]
        def diff(col):
            a, b = latest[col], prev[col]
            return float(a - b) if pd.notna(a) and pd.notna(b) else np.nan
        feats["plant_height_diff"]       = diff("plant_height")
        feats["growth_length_diff"]      = diff("growth_length")
        feats["cluster_num_diff"]        = diff("cluster_num")
        feats["fruits_per_cluster_diff"] = diff("fruits_per_cluster")

    return feats


# ── 6. snapshot 데이터셋 생성 ─────────────────────────────────
def make_snapshot_dataset(base_df, env_all, grow_all):
    print("2. snapshot 데이터셋 생성 중...")
    rows = []
    env_grouped  = {cid: g for cid, g in env_all.groupby("cult_id")}
    grow_grouped = {cid: g for cid, g in grow_all.groupby("cult_id")}

    for _, base in base_df.iterrows():
        cid           = base["cult_id"]
        planting_date = base["planting_date"]
        first_harvest = base["first_harvest_date"]
        cult_end      = base["cult_end_date"]
        area          = base["planting_area"]
        total_area    = base["total_area"]

        env_cult  = env_grouped.get(cid,  pd.DataFrame())
        grow_cult = grow_grouped.get(cid, pd.DataFrame())

        for snap_day in SNAPSHOT_DAYS:
            snapshot_date = planting_date + pd.Timedelta(days=snap_day)

            if pd.notna(first_harvest) and snapshot_date >= first_harvest:
                continue
            if pd.notna(cult_end) and snapshot_date > cult_end:
                continue

            ef = env_features(env_cult, snapshot_date)
            if not ef or ef.get("env_days", 0) < 7:
                continue   # 환경 데이터 7일 미만은 스킵

            gf = grow_features(grow_cult, snapshot_date)

            row = {
                "cult_id":        cid,
                "snapshot_day":   snap_day,
                "target":         float(base["yield_per_area"]),
                # 재배 기본
                "planting_area":    float(area),
                "planting_density": float(base["planting_density"]) if pd.notna(base["planting_density"]) else np.nan,
                "crop_cycle":       float(base["crop_cycle"])        if pd.notna(base["crop_cycle"])       else np.nan,
                "survey_year":      float(base["survey_year"])       if pd.notna(base["survey_year"])      else np.nan,
                "area_ratio":       float(area / total_area)         if pd.notna(total_area) and total_area > 0 else np.nan,
                "planting_month":   planting_date.month,
                # 카테고리
                "house_type":     base["house_type"],
                "house_form":     base["house_form"],
                "region_l1":      base["region_l1"],
                "region_l2":      base["region_l2"],
                "planting_season": season_from_month(planting_date.month),
            }
            row.update(ef)
            row.update(gf)
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"   - snapshot rows : {len(df):,}")
    print(f"   - unique cults  : {df['cult_id'].nunique():,}")
    return df.reset_index(drop=True)


# ── 7. 모델 학습 ──────────────────────────────────────────────
NUM_FEATURES = [
    "snapshot_day", "planting_area", "planting_density", "crop_cycle",
    "survey_year", "area_ratio", "planting_month",
    # 환경
    "env_days", "acc_temp", "acc_solar",
    "avg_in_temp", "avg_in_humidity", "avg_in_co2", "avg_acc_solar",
    "rain_days", "r7_in_temp", "r7_solar", "r14_in_temp", "r14_solar",
    # 생육
    "has_growth", "plant_height", "growth_length", "leaf_count",
    "cluster_num", "fruits_per_cluster", "branch_width",
    "plant_height_diff", "growth_length_diff",
    "cluster_num_diff", "fruits_per_cluster_diff", "grow_obs_days",
]
CAT_FEATURES = ["house_type", "house_form", "region_l1", "region_l2", "planting_season"]


def train_model(snap_df):
    print("3. 모델 학습 (target: yield_per_area = kg/평)")

    X = snap_df[NUM_FEATURES + CAT_FEATURES].copy()
    y = np.log1p(snap_df["target"].astype(float))
    groups = snap_df["cult_id"]

    all_null = [c for c in X.columns if X[c].notna().sum() == 0]
    if all_null:
        print(f"   null 전체 컬럼 제거: {all_null}")

    num_feats = [c for c in NUM_FEATURES if c not in all_null]
    cat_feats = [c for c in CAT_FEATURES if c not in all_null]
    X = X[num_feats + cat_feats].copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    print(f"   train: {len(X_tr):,} rows / {groups.iloc[train_idx].nunique()} cults")
    print(f"   test : {len(X_te):,} rows / {groups.iloc[test_idx].nunique()} cults")

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler())
        ]), num_feats),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_feats),
    ])

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(X_tr, y_tr)

    pred_log = pipeline.predict(X_te)
    y_true = np.expm1(y_te)
    pred   = np.maximum(np.expm1(pred_log), 0)

    r2  = r2_score(y_true, pred)
    mae = mean_absolute_error(y_true, pred)
    print(f"\n{'='*55}")
    print(f"R2  : {r2:.4f}")
    print(f"MAE : {mae:.4f} kg/평")
    print(f"{'='*55}")

    # 피처 중요도
    ohe_names = list(
        pipeline.named_steps["prep"]
        .named_transformers_["cat"]
        .named_steps["ohe"]
        .get_feature_names_out(cat_feats)
    )
    all_feat_names = num_feats + ohe_names
    imp = pd.DataFrame({"feature": all_feat_names,
                        "importance": pipeline.named_steps["model"].feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print("\n피처 중요도 TOP 20:")
    print(imp.head(20).to_string(index=False))

    # 오차 분석
    result_df = X_te.copy()
    result_df["cult_id"]  = groups.iloc[test_idx].values
    result_df["actual"]   = y_true.values
    result_df["pred"]     = pred
    result_df["abs_err"]  = np.abs(result_df["actual"] - result_df["pred"])
    print("\n오차 큰 상위 10건:")
    print(result_df.sort_values("abs_err", ascending=False)[
        ["cult_id","snapshot_day","actual","pred","abs_err"]].head(10))

    return pipeline, result_df, r2, mae, num_feats, cat_feats


# ── 8. 저장 ───────────────────────────────────────────────────
def save_artifacts(pipeline, result_df, r2, mae):
    model_dir = os.path.join(package_root, "smartfarm", "ml", "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "v3_yield_pipeline.joblib")
    image_path = os.path.join(model_dir, "v3_yield_result.png")
    csv_path   = os.path.join(model_dir, "v3_yield_result.csv")

    joblib.dump(pipeline, model_path)
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 8))
    plt.scatter(result_df["actual"], result_df["pred"], alpha=0.5, s=15)
    lim = (min(result_df["actual"].min(), result_df["pred"].min()),
           max(result_df["actual"].max(), result_df["pred"].max()))
    plt.plot(lim, lim, "--", linewidth=2)
    plt.xlabel("실제 (kg/평)")
    plt.ylabel("예측 (kg/평)")
    plt.title(f"생산량 V3 (환경+생육+재배) R2={r2:.4f} MAE={mae:.2f}kg/평")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    plt.close()

    print(f"\n저장 완료:")
    print(f"  모델: {model_path}")
    print(f"  이미지: {image_path}")
    print(f"  CSV: {csv_path}")


def main():
    app = create_app(enable_scheduler=False)
    with app.app_context():
        print("1. 기본 데이터 로드...")
        base_df = load_base()

        cult_ids = base_df["cult_id"].tolist()
        print(f"   환경 데이터 있는 cult: {len(cult_ids)}")

        env_all  = load_env_all(cult_ids)
        grow_all = load_grow_all(cult_ids)
        print(f"   env rows : {len(env_all):,}")
        print(f"   grow rows: {len(grow_all):,}")

        snap_df = make_snapshot_dataset(base_df, env_all, grow_all)

        pipeline, result_df, r2, mae, num_feats, cat_feats = train_model(snap_df)
        save_artifacts(pipeline, result_df, r2, mae)


if __name__ == "__main__":
    main()
