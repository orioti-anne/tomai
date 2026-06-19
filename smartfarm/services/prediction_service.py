from __future__ import annotations
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from sqlalchemy import text
from smartfarm import db
from smartfarm.ml.services.environment_recommendation_service import recommend_environment



load_dotenv()

def send_to_gcp(category: str, value: float, target_date: str = None):
    pass  # GCP 전송 제거됨


def run_ml_prediction_with_push(cult_id: int):
    return run_default_prediction(cult_id=cult_id)




BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "ml" / "models"

print("[MODEL_DIR]", MODEL_DIR)

# ---------------------------------------------------------
# 생산량 예측 서비스 기준값
# ---------------------------------------------------------
MIN_ENV_DAYS = 7
SNAPSHOT_DAYS = [14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98]


def run_environment_recommendation(sensor_data: dict) -> dict:
    return recommend_environment(sensor_data)


def run_ml_prediction(cult_id: int) -> Dict[str, Any]:
    return run_default_prediction(cult_id=cult_id)

def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default

def _load_model(candidates: list[str]):
    for name in candidates:
        path = MODEL_DIR / name
        print(f"[MODEL CHECK] {path} exists={path.exists()}")

        if path.exists():
            try:
                model = joblib.load(path)
                print(f"[MODEL LOAD OK] {path}")
                return model, str(path)
            except Exception as e:
                print(f"[MODEL LOAD ERROR] {path} / {type(e).__name__}: {e}")

    print("[MODEL LOAD FAIL] no model loaded")
    return None, None


# =========================================================
# 공통: 계절
# =========================================================
def _season_from_month(month: int | None) -> Optional[str]:
    if month is None:
        return None
    return {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall", 10: "fall", 11: "fall",
    }.get(int(month))


# =========================================================
# 가격 예측용 데이터
# =========================================================
def _get_recent_market_data() -> pd.DataFrame:
    try:
        query = text("""
                     SELECT p.price_date,
                            p.price_per_kg,
                            w.avg_temp
                     FROM kamis_tomato_price p
                              LEFT JOIN weather_index w ON p.price_date = w.w_date
                     WHERE p.item_name = '완숙토마토'
                       AND p.price_date >= CURRENT_DATE - INTERVAL '400 days'
                     ORDER BY p.price_date ASC
                     """)

        df = pd.read_sql(query, db.engine)
        if df.empty: return pd.DataFrame()

        df.columns = [str(c).lower() for c in df.columns]
        df["price_date"] = pd.to_datetime(df["price_date"])

        df = df.set_index("price_date").resample('D').asfreq().reset_index()
        df["price_per_kg"] = df["price_per_kg"].ffill().bfill()
        df["avg_temp"] = df["avg_temp"].ffill().bfill()

        return df
    except Exception as e:
        print(f"[MARKET DATA ERROR] {e}")
        return pd.DataFrame()


def _get_latest_market_price() -> float:
    try:
        query = text("""
                     SELECT price_per_kg
                     FROM kamis_tomato_price
                     WHERE item_name = '완숙토마토'
                     ORDER BY price_date DESC
                     LIMIT 1
                     """)

        df = pd.read_sql(query, db.engine)

        if not df.empty:
            df.columns = [str(c).lower() for c in df.columns]
            return _to_float(df.iloc[0]["price_per_kg"], 3500.0)

    except Exception as e:
        print(f"[LATEST PRICE ERROR] {type(e).__name__}: {e}")

    return 3500.0


# =========================================================
# 생산량 예측용 snapshot feature 생성
# =========================================================
def _get_base_cultivation_row(cult_id: int) -> Optional[dict]:
    try:
        query = text("""
                     SELECT c.cult_id,
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
                              JOIN farms f
                                   ON c.farm_id = f.farm_id
                              LEFT JOIN prod_summary ps
                                        ON c.cult_id = ps.cult_id
                              LEFT JOIN products p
                                        ON c.cult_id = p.cult_id
                     WHERE c.cult_id = :cult_id
                     GROUP BY c.cult_id,
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
                              ps.cult_total_quantity
                     """)

        df = pd.read_sql(query, db.engine, params={"cult_id": cult_id})
        if df.empty:
            return None

        row = df.iloc[0].to_dict()
        for col in ["planting_date", "first_harvest_date", "cult_end_date"]:
            if row.get(col) is not None:
                row[col] = pd.to_datetime(row[col], errors="coerce")
        return row

    except Exception as e:
        print(f"[BASE CULT ERROR] cult_id={cult_id}, {type(e).__name__}: {e}")
        return None


def _get_env_slice(cult_id: int, planting_date: pd.Timestamp, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    query = text("""
                 SELECT cult_id,
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
                 WHERE cult_id = :cult_id
                   AND measure_date >= :planting_date
                   AND measure_date <= :snapshot_date
                 ORDER BY measure_date
                 """)

    df = pd.read_sql(
        query,
        db.engine,
        params={
            "cult_id": cult_id,
            "planting_date": planting_date.to_pydatetime(),
            "snapshot_date": snapshot_date.to_pydatetime(),
        },
    )
    if not df.empty:
        df["measure_date"] = pd.to_datetime(df["measure_date"], errors="coerce")
    return df


def _get_growth_slice(cult_id: int, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    query = text("""
                 SELECT cult_id,
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
                 WHERE cult_id = :cult_id
                   AND inspect_date <= :snapshot_date
                 ORDER BY inspect_date
                 """)

    df = pd.read_sql(
        query,
        db.engine,
        params={
            "cult_id": cult_id,
            "snapshot_date": snapshot_date.to_pydatetime(),
        },
    )
    if not df.empty:
        df["inspect_date"] = pd.to_datetime(df["inspect_date"], errors="coerce")
    return df


def _get_recent_env_features(env_slice: pd.DataFrame, snapshot_date: pd.Timestamp) -> dict:
    if env_slice.empty:
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


def _get_growth_features(growth_slice: pd.DataFrame) -> dict:
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
    }

    if growth_slice.empty:
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

        result.update({
            "plant_height_diff": latest.get("plant_height") - prev.get("plant_height")
            if pd.notna(latest.get("plant_height")) and pd.notna(prev.get("plant_height")) else np.nan,
            "growth_length_diff": latest.get("growth_length") - prev.get("growth_length")
            if pd.notna(latest.get("growth_length")) and pd.notna(prev.get("growth_length")) else np.nan,
            "leaf_count_diff": latest.get("leaf_count") - prev.get("leaf_count")
            if pd.notna(latest.get("leaf_count")) and pd.notna(prev.get("leaf_count")) else np.nan,
            "cluster_num_diff": latest.get("cluster_num") - prev.get("cluster_num")
            if pd.notna(latest.get("cluster_num")) and pd.notna(prev.get("cluster_num")) else np.nan,
            "fruits_per_cluster_diff": latest.get("fruits_per_cluster") - prev.get("fruits_per_cluster")
            if pd.notna(latest.get("fruits_per_cluster")) and pd.notna(prev.get("fruits_per_cluster")) else np.nan,
        })

    return result


def _nearest_snapshot_day(day: int) -> int:
    return min(SNAPSHOT_DAYS, key=lambda x: abs(x - day))


def _build_yield_feature_row(cult_id: int) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    base = _get_base_cultivation_row(cult_id)
    if not base:
        print(f"[YIELD FEATURE] no base row for cult_id={cult_id}")
        return None, "재배 기본정보를 찾을 수 없습니다."

    planting_date = pd.to_datetime(base.get("planting_date"), errors="coerce")
    if pd.isna(planting_date):
        print(f"[YIELD FEATURE] planting_date missing cult_id={cult_id}")
        return None, "정식일 정보가 없어 예측 모델을 실행할 수 없습니다."

    today = pd.Timestamp(date.today())
    snapshot_date = today
    snapshot_day_raw = max(0, (snapshot_date.date() - planting_date.date()).days)

    first_harvest_date = pd.to_datetime(base.get("first_harvest_date"), errors="coerce")
    if pd.notna(first_harvest_date) and snapshot_date >= first_harvest_date:
        snapshot_date = first_harvest_date - pd.Timedelta(days=1)
        snapshot_day_raw = max(0, (snapshot_date.date() - planting_date.date()).days)

    snapshot_day = _nearest_snapshot_day(snapshot_day_raw)

    env_slice = _get_env_slice(cult_id, planting_date, snapshot_date)
    env_days = len(env_slice)

    print(
        f"[YIELD FEATURE] cult_id={cult_id}, "
        f"snapshot_day_raw={snapshot_day_raw}, snapshot_day={snapshot_day}, env_days={env_days}"
    )

    if env_slice.empty:
        print(f"[YIELD FEATURE] env_slice empty cult_id={cult_id}", f"required={MIN_ENV_DAYS}")
        return None, f"환경데이터가 없어 예상 생산량은 임시 기준값으로 계산되었습니다. 최소 {MIN_ENV_DAYS}일 이상의 환경데이터가 필요합니다."

    if env_days < MIN_ENV_DAYS:
        print(
            f"[YIELD FEATURE] insufficient env data cult_id={cult_id}, "
            f"env_days={env_days}, required={MIN_ENV_DAYS}"
        )
        return None, f"환경데이터가 {env_days}일만 누적되어 있어 예상 생산량은 임시 기준값으로 계산되었습니다. 최소 {MIN_ENV_DAYS}일 이상의 환경데이터가 필요합니다."

    latest_env = env_slice.sort_values("measure_date").iloc[-1]
    recent_env = _get_recent_env_features(env_slice, snapshot_date)
    growth_slice = _get_growth_slice(cult_id, snapshot_date)
    growth_feat = _get_growth_features(growth_slice)

    row = {
        "crop_cycle": _to_float(base.get("crop_cycle")),
        "planting_area": _to_float(base.get("planting_area")),
        "planting_density": _to_float(base.get("planting_density")),
        "survey_year": _to_float(base.get("survey_year")),
        "total_area": _to_float(base.get("total_area")),
        "farm_num": _to_float(base.get("farm_num")),
        "first_survey_year": _to_float(base.get("first_survey_year")),
        "planting_month": planting_date.month,
        "snapshot_day": snapshot_day,
        "area_ratio": (
            _to_float(base.get("planting_area")) / _to_float(base.get("total_area"))
            if _to_float(base.get("planting_area")) and _to_float(base.get("total_area")) and _to_float(
                base.get("total_area")) > 0
            else np.nan
        ),
        "env_days": env_days,
        "acc_temp_to_snapshot": _to_float(latest_env.get("acc_temp")),
        "acc_solar_to_snapshot": _to_float(latest_env.get("acc_solar")),
        "avg_daily_out_temp": pd.to_numeric(env_slice["daily_out_temp"], errors="coerce").mean(),
        "avg_daily_in_temp": pd.to_numeric(env_slice["daily_in_temp"], errors="coerce").mean(),
        "avg_daily_in_humidity": pd.to_numeric(env_slice["daily_in_humidity"], errors="coerce").mean(),
        "avg_daily_in_co2": pd.to_numeric(env_slice["daily_in_co2"], errors="coerce").mean(),
        "avg_daily_soil_temp": pd.to_numeric(env_slice["daily_soil_temp"], errors="coerce").mean(),
        "avg_daily_acc_solar": pd.to_numeric(env_slice["daily_acc_solar"], errors="coerce").mean(),
        "rain_days_total": pd.to_numeric(env_slice["daily_rain_detection"], errors="coerce").fillna(0).sum(),
        "item": base.get("item"),
        "item_variety": base.get("item_variety"),
        "house_type": base.get("house_type"),
        "house_form": base.get("house_form"),
        "region_l1": base.get("region_l1"),
        "region_l2": base.get("region_l2"),
        "planting_season": _season_from_month(planting_date.month),
    }

    row.update(recent_env)
    row.update(growth_feat)

    X_y = pd.DataFrame([row])
    print("[YIELD FEATURE ROW]")
    print(X_y.T)
    return X_y, None


# =========================================================
# V3 생산량 feature 빌더 (환경 + 생육 + 재배 기본정보)
# =========================================================
def _build_v3_yield_feature_row(cult_id: int, snapshot_day: int) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    base = _get_base_cultivation_row(cult_id)
    if not base:
        return None, "재배 기본정보 없음"

    planting_date = pd.to_datetime(base.get("planting_date"), errors="coerce")
    if pd.isna(planting_date):
        return None, "정식일 없음"

    today = pd.Timestamp(date.today())
    snapshot_date = planting_date + pd.Timedelta(days=snapshot_day)
    if snapshot_date > today:
        snapshot_date = today

    env_slice    = _get_env_slice(cult_id, planting_date, snapshot_date)
    growth_slice = _get_growth_slice(cult_id, snapshot_date)

    env_days = len(env_slice)
    if env_days < MIN_ENV_DAYS:
        return None, f"환경데이터 부족 ({env_days}일)"

    # 환경 집계
    latest_env = env_slice.sort_values("measure_date").iloc[-1]
    r7  = env_slice[env_slice["measure_date"] > snapshot_date - pd.Timedelta(days=7)]
    r14 = env_slice[env_slice["measure_date"] > snapshot_date - pd.Timedelta(days=14)]

    def safe_mean(s): return float(pd.to_numeric(s, errors="coerce").mean())

    area       = _to_float(base.get("planting_area"))
    total_area = _to_float(base.get("total_area"))

    row = {
        # 재배 기본
        "snapshot_day":      snapshot_day,
        "planting_area":     area,
        "planting_density":  _to_float(base.get("planting_density")),
        "crop_cycle":        _to_float(base.get("crop_cycle")),
        "survey_year":       _to_float(base.get("survey_year")),
        "area_ratio":        area / total_area if area and total_area and total_area > 0 else np.nan,
        "planting_month":    planting_date.month,
        # 환경
        "env_days":          env_days,
        "acc_temp":          _to_float(latest_env.get("acc_temp")),
        "acc_solar":         _to_float(latest_env.get("acc_solar")),
        "avg_in_temp":       safe_mean(env_slice["daily_in_temp"]),
        "avg_in_humidity":   safe_mean(env_slice["daily_in_humidity"]),
        "avg_in_co2":        safe_mean(env_slice["daily_in_co2"]),
        "avg_acc_solar":     safe_mean(env_slice["daily_acc_solar"]),
        "rain_days":         float(pd.to_numeric(env_slice["daily_rain_detection"], errors="coerce").fillna(0).sum()),
        "r7_in_temp":        safe_mean(r7["daily_in_temp"])  if len(r7) > 0 else np.nan,
        "r7_solar":          safe_mean(r7["daily_acc_solar"]) if len(r7) > 0 else np.nan,
        "r14_in_temp":       safe_mean(r14["daily_in_temp"]) if len(r14) > 0 else np.nan,
        "r14_solar":         safe_mean(r14["daily_acc_solar"]) if len(r14) > 0 else np.nan,
        # 카테고리
        "house_type":        base.get("house_type"),
        "house_form":        base.get("house_form"),
        "region_l1":         base.get("region_l1"),
        "region_l2":         base.get("region_l2"),
        "planting_season":   _season_from_month(planting_date.month),
    }

    # 생육 집계 (inspect_date별 plant 평균 → 최신 inspect_date)
    has_growth = 0
    gf = {
        "has_growth": 0,
        "plant_height": np.nan, "growth_length": np.nan, "leaf_count": np.nan,
        "cluster_num": np.nan, "fruits_per_cluster": np.nan, "branch_width": np.nan,
        "plant_height_diff": np.nan, "growth_length_diff": np.nan,
        "cluster_num_diff": np.nan, "fruits_per_cluster_diff": np.nan,
        "grow_obs_days": np.nan,
    }
    if not growth_slice.empty:
        # plant_num별 평균 → inspect_date별 1행
        grow_cols = ["plant_height", "growth_length", "leaf_count",
                     "cluster_num", "fruits_per_cluster", "branch_width"]
        agg = growth_slice.groupby("inspect_date")[grow_cols].mean().reset_index()
        agg = agg.sort_values("inspect_date")

        latest_g = agg.iloc[-1]
        gf["has_growth"] = 1
        for c in grow_cols:
            gf[c] = float(latest_g[c]) if pd.notna(latest_g[c]) else np.nan
        gf["grow_obs_days"] = float((agg["inspect_date"].iloc[-1] - agg["inspect_date"].iloc[0]).days) if len(agg) > 1 else 0.0

        if len(agg) >= 2:
            prev_g = agg.iloc[-2]
            def diff(col): return float(latest_g[col] - prev_g[col]) if pd.notna(latest_g[col]) and pd.notna(prev_g[col]) else np.nan
            gf["plant_height_diff"]       = diff("plant_height")
            gf["growth_length_diff"]      = diff("growth_length")
            gf["cluster_num_diff"]        = diff("cluster_num")
            gf["fruits_per_cluster_diff"] = diff("fruits_per_cluster")

    row.update(gf)
    return pd.DataFrame([row]), None


def _get_expected_cult_duration(crop_cycle: Optional[float], fallback: float = 284.0, min_samples: int = 5) -> float:
    """
    동일 crop_cycle 의 완료 재배 평균 기간을 반환.
    샘플 수 부족 시: 바로 아래 작기(crop_cycle - 1)부터 내려가며 탐색.
    모두 부족하면 fallback(학습 중앙값) 사용.
    """
    if crop_cycle is None:
        return fallback
    try:
        cc = int(crop_cycle)
        # 요청 작기부터 1까지 내려가며 n >= min_samples인 첫 번째 값 사용
        for candidate_cc in range(cc, 0, -1):
            row = db.session.execute(text("""
                SELECT ROUND(AVG((ps.cult_end_date - c.planting_date)::int)), COUNT(*)
                FROM cultivations c
                JOIN prod_summary ps ON c.cult_id = ps.cult_id
                WHERE ps.cult_total_quantity > 0
                  AND c.planting_date IS NOT NULL
                  AND ps.cult_end_date IS NOT NULL
                  AND (ps.cult_end_date - c.planting_date) BETWEEN 60 AND 600
                  AND c.crop_cycle = :cc
            """), {"cc": candidate_cc}).fetchone()
            if row and row[0] is not None and row[1] >= min_samples:
                if candidate_cc != cc:
                    print(f"[CULT_DURATION] crop_cycle={cc} 샘플 부족 → crop_cycle={candidate_cc} 평균 사용: {row[0]}일 (n={row[1]}건)")
                return float(row[0])
    except Exception:
        pass
    return fallback


# =========================================================
# V4 생산량 feature 빌더 (재배 전체 기간 집계 — train_yield_model_v4 와 동일한 로직)
# =========================================================
def _build_v4_yield_feature_row(cult_id: int, bundle: dict, snap_day: Optional[int] = None) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[float]]:
    base = _get_base_cultivation_row(cult_id)
    if not base:
        return None, "재배 기본정보 없음", None

    planting_date = pd.to_datetime(base.get("planting_date"), errors="coerce")
    if pd.isna(planting_date):
        return None, "정식일 없음", None

    today_ts = pd.Timestamp(date.today())
    env_df   = _get_env_slice(cult_id, planting_date, today_ts)
    grow_df  = _get_growth_slice(cult_id, today_ts)

    has_env = len(env_df) >= MIN_ENV_DAYS
    if not has_env:
        print(f"[YIELD V4] cult_id={cult_id} 환경데이터 부족({len(env_df)}일) — 훈련 중앙값으로 대체")

    # ── 환경 피처 (데이터 있으면 통계, 없으면 NaN → imputer가 중앙값 대체) ──
    if has_env:
        env_df = env_df.sort_values("measure_date")
        total_days = (env_df["measure_date"].max() - env_df["measure_date"].min()).days + 1
    else:
        total_days = np.nan

    def _estats(col):
        if not has_env:
            return {f"{col}_mean": np.nan, f"{col}_std": np.nan,
                    f"{col}_min": np.nan, f"{col}_max": np.nan}
        s = pd.to_numeric(env_df[col], errors="coerce").dropna()
        if len(s) == 0:
            return {f"{col}_mean": np.nan, f"{col}_std": np.nan,
                    f"{col}_min": np.nan, f"{col}_max": np.nan}
        return {f"{col}_mean": s.mean(), f"{col}_std": s.std(),
                f"{col}_min": s.min(),  f"{col}_max": s.max()}

    env_feats = {"env_total_days": total_days}
    for col in ["daily_in_temp", "daily_in_humidity", "daily_in_co2", "daily_acc_solar"]:
        env_feats.update(_estats(col))

    if has_env:
        temp = pd.to_numeric(env_df["daily_in_temp"], errors="coerce")
        env_feats["heat_days"]  = int((temp > 30).sum())
        env_feats["cold_days"]  = int((temp < 10).sum())
    else:
        env_feats["heat_days"]  = np.nan
        env_feats["cold_days"]  = np.nan
    if has_env:
        env_feats["rain_days"] = int(pd.to_numeric(env_df["daily_rain_detection"], errors="coerce").fillna(0).sum())
        last_env = env_df.dropna(subset=["acc_temp"]).iloc[-1] if env_df["acc_temp"].notna().any() else None
        env_feats["acc_temp_final"]  = float(last_env["acc_temp"])  if last_env is not None else np.nan
        env_feats["acc_solar_final"] = float(last_env["acc_solar"]) if last_env is not None else np.nan
        if total_days > 30:
            late   = env_df[env_df["measure_date"] >= env_df["measure_date"].max() - pd.Timedelta(days=30)]
            t_late = pd.to_numeric(late["daily_in_temp"], errors="coerce").dropna()
            env_feats["late30_temp_mean"] = t_late.mean() if len(t_late) > 0 else np.nan
        else:
            env_feats["late30_temp_mean"] = np.nan
    else:
        env_feats["rain_days"]       = np.nan
        env_feats["acc_temp_final"]  = np.nan
        env_feats["acc_solar_final"] = np.nan
        env_feats["late30_temp_mean"] = np.nan

    # ── 생육 피처 (날짜별 평균 → 시계열 통계) ───────────────
    GROW_COLS = ["plant_height", "growth_length", "leaf_count", "cluster_num",
                 "fruits_per_cluster", "branch_width", "flowers_per_cluster", "blooming_per_cluster"]
    grow_feats = {"grow_n_dates": 0, "grow_span_days": 0}
    for c in GROW_COLS:
        grow_feats.update({f"{c}_mean": np.nan, f"{c}_std": np.nan,
                           f"{c}_final": np.nan, f"{c}_slope": np.nan})

    if not grow_df.empty:
        for c in GROW_COLS:
            grow_df[c] = pd.to_numeric(grow_df[c], errors="coerce")
        daily = grow_df.groupby("inspect_date")[GROW_COLS].mean().reset_index().sort_values("inspect_date")
        n_dates = len(daily)
        grow_feats["grow_n_dates"] = n_dates
        if n_dates >= 2:
            grow_feats["grow_span_days"] = (daily["inspect_date"].iloc[-1] - daily["inspect_date"].iloc[0]).days

        for col in GROW_COLS:
            s = daily[col].dropna()
            if len(s) == 0:
                continue
            grow_feats[f"{col}_mean"]  = s.mean()
            grow_feats[f"{col}_std"]   = s.std()
            grow_feats[f"{col}_final"] = float(daily[col].dropna().iloc[-1])
            valid = daily[["inspect_date", col]].dropna()
            if len(valid) >= 3:
                x = (valid["inspect_date"] - valid["inspect_date"].min()).dt.days.values
                grow_feats[f"{col}_slope"] = np.polyfit(x, valid[col].values, 1)[0]

    # ── 기본 재배 피처 ──────────────────────────────────────
    area       = _to_float(base.get("planting_area"))
    total_area = _to_float(base.get("total_area"))
    pm         = planting_date.month

    # cult_duration: 완료된 재배는 실제 기간, 진행 중이면 추천 출하일 사용
    cult_end = base.get("cult_end_date")
    if pd.notna(cult_end):
        cult_duration = float((pd.to_datetime(cult_end) - planting_date).days)
    else:
        # 진행 중인 재배: 동일 작기 기준 평균 전체 재배기간 사용
        # (95/105/115일은 가격 비교 시점이지 전체 재배기간이 아님)
        crop_cycle_val = _to_float(base.get("crop_cycle"))
        cult_duration = _get_expected_cult_duration(crop_cycle_val)

    # v5: snap_day가 주어지면 cult_duration을 시나리오 일수로 오버라이드
    if snap_day is not None:
        cult_duration = float(snap_day)

    row = {
        "planting_area":    area,
        "planting_density": _to_float(base.get("planting_density")),
        "crop_cycle":       _to_float(base.get("crop_cycle")),
        "survey_year":      _to_float(base.get("survey_year")) or float(date.today().year),
        "area_ratio":       area / total_area if area and total_area and total_area > 0 else np.nan,
        "planting_month":   float(pm),
        "cult_duration":    cult_duration,
        "house_type":       base.get("house_type"),
        "house_form":       base.get("house_form"),
        "region_l1":        base.get("region_l1"),
        "planting_season":  _season_from_month(pm),
    }
    row.update(env_feats)
    row.update(grow_feats)

    num_feats = bundle.get("num_features", [])
    cat_feats = bundle.get("cat_features", [])
    for f in num_feats:
        if f not in row:
            row[f] = np.nan
    for f in cat_feats:
        if f not in row:
            row[f] = None

    return pd.DataFrame([row])[num_feats + cat_feats], None, cult_duration


# =========================================================
# 간단 생산량 feature 빌더 (env/growth 없이 재배 기본 정보만)
# =========================================================
def _build_simple_yield_feature_row(cult_id: int, snapshot_day: int) -> Optional[pd.DataFrame]:
    base = _get_base_cultivation_row(cult_id)
    if not base:
        return None

    planting_date = pd.to_datetime(base.get("planting_date"), errors="coerce")
    if pd.isna(planting_date):
        return None

    survey_year = _to_float(base.get("survey_year")) or float(planting_date.year)
    planting_area = _to_float(base.get("planting_area"))
    total_area = _to_float(base.get("total_area"))

    row = {
        "crop_cycle": _to_float(base.get("crop_cycle")),
        "planting_area": planting_area,
        "planting_density": _to_float(base.get("planting_density")),
        "survey_year": survey_year,
        "total_area": total_area,
        "farm_num": _to_float(base.get("farm_num")),
        "first_survey_year": _to_float(base.get("first_survey_year")),
        "planting_month": planting_date.month,
        "snapshot_day": snapshot_day,
        "area_ratio": (
            planting_area / total_area
            if planting_area and total_area and total_area > 0
            else np.nan
        ),
        "item": base.get("item"),
        "item_variety": base.get("item_variety"),
        "house_type": base.get("house_type"),
        "house_form": base.get("house_form"),
        "region_l1": base.get("region_l1"),
        "region_l2": base.get("region_l2"),
        "planting_season": _season_from_month(planting_date.month),
    }

    return pd.DataFrame([row])


# =========================================================
# fallback 결과 생성
# =========================================================
def _build_fallback_result(
        *,
        area: float,
        days_passed: int,
        final_price: float,
        curr_price: float,
        message: str,
        reason_code: str,
) -> Dict[str, Any]:
    harvest_target = 105
    days_to_go = max(0, harvest_target - days_passed)
    expected_date = (date.today() + timedelta(days=days_to_go)).isoformat()

    final_yield_total = area * 30.0 if area > 0 else 0.0
    print(f"[YIELD FALLBACK APPLIED] final_yield_total={final_yield_total}, reason={reason_code}")

    result = {
        "expected_harvest_date": expected_date,
        "harvest_status": "finished" if days_passed > 130 else "active",
        "avg_days_to_peak_harvest": days_to_go,
        "avg_yield_per_m2": round(final_yield_total / area, 2) if area and area > 0 else 0,
        "expected_quantity": round(final_yield_total, 1),
        "expected_price_per_kg": round(final_price, 1),
        "expected_sales": round(final_yield_total * final_price, 0),
        "sample_count": 0,
        "price_feature_row": {
            "PREV_PER_KG_1D": round(curr_price, 1)
        },
        "prediction_source": "fallback",
        "prediction_confidence": "low",
        "prediction_message": message,
        "yield_fallback_reason": reason_code,
    }

    print("[FINAL RESULT]", result)
    return result


# =========================================================
# 기본 예측
# =========================================================
def run_default_prediction(**kwargs) -> Dict[str, Any]:
    cult_id = kwargs.get("cult_id")
    area = _to_float(kwargs.get("planting_area"), 0.0)
    p_date_str = kwargs.get("planting_date")

    # cult_id만 전달된 경우 DB에서 기본 재배정보 보완
    if cult_id and (not area or area <= 0 or not p_date_str):
        _base = _get_base_cultivation_row(cult_id)
        if _base:
            if not area or area <= 0:
                area = _to_float(_base.get("planting_area"), 0.0)
            if not p_date_str:
                _pd = _base.get("planting_date")
                if _pd:
                    p_date_str = str(_pd)[:10]

    try:
        p_date = datetime.strptime(p_date_str, "%Y-%m-%d").date() if p_date_str else date.today()
    except:
        p_date = date.today()

    days_passed = (date.today() - p_date).days

    # 1. 모델 및 기초 데이터 준비
    model_p,         path_p  = _load_model(["v7_tomato_price_pipelines.joblib", "v5_tomato_price_pipeline.joblib", "v4_tomato_price_pipeline.joblib"])
    model_v4_bundle, path_v4 = _load_model(["v5_yield_pipeline.joblib", "v4_yield_pipeline.joblib"])
    df_market = _get_recent_market_data()
    latest_price = _get_latest_market_price()

    evaluation_results = []

    # 폴백 발생 여부를 추적하기 위한 변수
    is_fallback = False
    fallback_reason = None
    fallback_message = "재배 및 환경데이터를 기반으로 계산되었습니다."

    # --- [B] 시나리오 분기: expected_cult_duration으로 candidates 결정 ---
    yield_reason = None
    expected_cult_duration = 284.0
    if cult_id:
        try:
            _, _, _dur = _build_v4_yield_feature_row(int(cult_id), model_v4_bundle or {})
            if _dur is not None:
                expected_cult_duration = _dur
        except Exception:
            pass

    if expected_cult_duration > 200:
        candidates = [180, 220, 260]
    else:
        candidates = [100, 125, 150]
    print(f"[CANDIDATES] cult_duration={expected_cult_duration:.0f}일 → 시나리오={candidates}")

    # 2. 루프 시작 (가격 시나리오별)
    for days in candidates:
        target_date = datetime.combine(p_date + timedelta(days=days), datetime.min.time())

        # --- [A] 가격 예측 ---
        ma30_price = latest_price
        if not df_market.empty:
            ma30_price = float(df_market["price_per_kg"].tail(30).mean())
            if ma30_price <= 0:
                ma30_price = latest_price

        final_price = ma30_price
        price_pipe = model_p.get(days) if isinstance(model_p, dict) else model_p
        if price_pipe and not df_market.empty:
            try:
                price_s = df_market["price_per_kg"]
                temp_s = df_market["avg_temp"]
                yoy_raw = float(price_s.iloc[-1]) / float(price_s.iloc[-365]) if len(price_s) >= 365 and float(price_s.iloc[-365]) > 0 else 1.0
                lag90  = float(price_s.iloc[-90])  if len(price_s) >= 90  else float(price_s.iloc[0])
                lag180 = float(price_s.iloc[-180]) if len(price_s) >= 180 else float(price_s.iloc[0])
                ma30   = float(price_s.tail(30).mean())
                ma90   = float(price_s.tail(90).mean())
                curr_month  = date.today().month
                target_month = target_date.month
                target_doy   = target_date.timetuple().tm_yday
                price_trend  = (ma30 - ma90) / ma90 if ma90 > 0 else 0.0
                input_p = pd.DataFrame([{
                    "MA_30D": ma30,
                    "MA_90D": ma90,
                    "YOY_RATIO": min(max(yoy_raw, 0.5), 1.5),
                    "GDD_30D": (temp_s.tail(30) - 10).clip(lower=0).sum(),
                    "TEMP_MA_30D": temp_s.tail(30).mean(),
                    "PRICE_LAG_90D": lag90,
                    "PRICE_LAG_180D": lag180,
                    "CURRENT_MONTH_SIN": np.sin(2 * np.pi * curr_month / 12),
                    "CURRENT_MONTH_COS": np.cos(2 * np.pi * curr_month / 12),
                    "TARGET_MONTH_SIN": np.sin(2 * np.pi * target_month / 12),
                    "TARGET_MONTH_COS": np.cos(2 * np.pi * target_month / 12),
                    "TARGET_DOY_SIN":  np.sin(2 * np.pi * target_doy / 365),
                    "TARGET_DOY_COS":  np.cos(2 * np.pi * target_doy / 365),
                    "PRICE_MA_7D":     float(price_s.tail(7).mean()),
                    "PRICE_TREND_30D": price_trend,
                }])
                raw_pred = price_pipe.predict(input_p)[0]
                predicted = float(np.expm1(raw_pred))
                if predicted >= 500:
                    final_price = predicted
                else:
                    print(f"[PRICE MODEL] raw={raw_pred:.4f} → {predicted:.1f}원, 범위 외 — MA30 사용: {ma30_price:.0f}원")
            except Exception as price_e:
                print(f"[PRICE MODEL ERROR] {type(price_e).__name__}: {price_e}")

        # --- [B] 수확량 예측 (시나리오 일수별 누적 수확량) ---
        snap_yield_total = area * 30.0 if area > 0 else 0.0  # 폴백
        snap_yield_reason = "model_unavailable"
        if cult_id and model_v4_bundle and isinstance(model_v4_bundle, dict):
            yield_pipe = model_v4_bundle.get("model")
            if yield_pipe:
                try:
                    X_y, y_err, _ = _build_v4_yield_feature_row(int(cult_id), model_v4_bundle, snap_day=days)
                    if X_y is not None:
                        per_area = float(yield_pipe.predict(X_y)[0])
                        if per_area > 0 and area > 0:
                            snap_yield_total  = per_area * area
                            snap_yield_reason = None
                            print(f"[YIELD V5] +{days}일 per_area={per_area:.2f} kg/평, total={snap_yield_total:.1f} kg")
                        else:
                            snap_yield_reason = "nonpositive"
                    else:
                        snap_yield_reason = "feature_unavailable"
                        print(f"[YIELD SKIP] {y_err}")
                except Exception as ye:
                    snap_yield_reason = "predict_error"
                    print(f"[YIELD ERROR] {type(ye).__name__}: {ye}")

        evaluation_results.append({
            "days": days,
            "date": target_date,
            "price": final_price,
            "yield": snap_yield_total,
            "expected_sales": snap_yield_total * final_price,
            "reason": snap_yield_reason
        })

    # 3. 최적 시점 결정
    best_option = max(evaluation_results, key=lambda x: x['expected_sales'])
    is_fallback   = (best_option["reason"] is not None)
    fallback_reason  = best_option["reason"]

    # 4. 결과 반환
    days_to_go = max(0, best_option["days"] - days_passed)

    final_is_fallback = (best_option["reason"] is not None)

    result = {
        "expected_harvest_date": best_option["date"].date().isoformat(),
        "harvest_status": "finished" if days_passed > 130 else "active",
        "avg_days_to_peak_harvest": days_to_go,
        "recommended_days_post_planting": best_option["days"],
        "avg_yield_per_m2": round(best_option["yield"] / area, 2) if area > 0 else 0,
        "expected_quantity": round(best_option["yield"], 1),
        "expected_price_per_kg": round(best_option["price"], 1),
        "expected_sales": round(best_option["expected_sales"], 0),
        "latest_market_price": latest_price,
        "comparison_data": [
            {"days": r["days"], "price": round(r["price"], 0), "yield": round(r["yield"], 1)}
            for r in evaluation_results
        ],
        # 하위 호환 필드 (DB 컬럼 매핑용, 실제 days는 comparison_data 참조)
        "price_day_95":  round(evaluation_results[0]["price"], 0) if len(evaluation_results) > 0 else 0,
        "price_day_105": round(evaluation_results[1]["price"], 0) if len(evaluation_results) > 1 else 0,
        "price_day_115": round(evaluation_results[2]["price"], 0) if len(evaluation_results) > 2 else 0,
        "prediction_source": "fallback" if final_is_fallback else "optimized_model",
        "prediction_confidence": "low" if final_is_fallback else "medium",
        "prediction_message": fallback_message if final_is_fallback else f"수익 분석 결과, 정식 후 {best_option['days']}일째 출하를 추천합니다.",
        "yield_fallback_reason": best_option["reason"]

    }

    print(f"[FINAL RESULT] Fallback: {final_is_fallback}, Reason: {best_option['reason']}")
    return result

def _build_yield_feature_row_at_day(cult_id, target_days):
    X_y, err = _build_yield_feature_row(cult_id)
    if X_y is not None:
        X_y["snapshot_day"] = _nearest_snapshot_day(target_days)
    return X_y, err