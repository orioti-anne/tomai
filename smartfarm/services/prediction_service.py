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
from smartfarm.ml.pipelines.tomato_pipeline import predict_tomato_cycle
from smartfarm.ml.services.environment_recommendation_service import recommend_environment



load_dotenv()

GCP_IP = os.getenv("GCP_IP")
if not GCP_IP:
    raise ValueError("환경 변수 GCP_IP가 설정되지 않았습니다. .env 파일을 확인하세요.")

GCP_ENDPOINT = f"http://{GCP_IP}:5000/api/receive-prediction"


def send_to_gcp(category: str, value: float, target_date: str = None):
    payload = {
        "type": category,
        "value": value,
        "target_date": target_date or datetime.now().strftime('%Y-%m-%d')
    }

    try:
        response = requests.post(GCP_ENDPOINT, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"🚀 [GCP 전송 성공] {category}: {value}")
        else:
            print(f"⚠️ [GCP 전송 오류] 상태코드: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ [GCP 전송 실패] 네트워크 연결 확인 필요: {e}")
    except Exception as e:
        print(f"❌ [GCP 전송 실패] 알 수 없는 오류: {e}")


def run_ml_prediction_with_push(cult_id: int):
    result = predict_tomato_cycle(cult_id=cult_id)
    # 계산된 결과를 GCP로 쏜다!
    send_to_gcp("growth_rgr", result.get('expected_quantity'))
    return result




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
    result = predict_tomato_cycle(cult_id=cult_id)

    if result and 'expected_quantity' in result:
        qty = result['expected_quantity']
        target_date = result.get('expected_harvest_date', datetime.now().strftime('%Y-%m-%d'))

        send_to_gcp("growth_prediction", qty, target_date)

    return result

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
                     SELECT P.PRICE_DATE,
                            P.PRICE_PER_KG,
                            W.AVG_TEMP
                     FROM KAMIS_TOMATO_PRICE P
                              LEFT JOIN WEATHER_INDEX W ON P.PRICE_DATE = W.W_DATE
                     WHERE P.ITEM_NAME = '완숙토마토'
                       AND P.PRICE_DATE >= TRUNC(SYSDATE) - 400
                     ORDER BY P.PRICE_DATE ASC
                     """)

        df = pd.read_sql(query, db.engine)
        if df.empty: return pd.DataFrame()

        df.columns = [str(c).upper() for c in df.columns]
        df["PRICE_DATE"] = pd.to_datetime(df["PRICE_DATE"])

        df = df.set_index("PRICE_DATE").resample('D').asfreq().reset_index()
        df["price_per_kg"] = df["price_per_kg"].ffill().bfill()
        df["AVG_TEMP"] = df["AVG_TEMP"].ffill().bfill()

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
            df.columns = [str(c).upper() for c in df.columns]
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

    try:
        p_date = datetime.strptime(p_date_str, "%Y-%m-%d").date() if p_date_str else date.today()
    except:
        p_date = date.today()

    days_passed = (date.today() - p_date).days

    # 1. 모델 및 기초 데이터 준비
    model_p, path_p = _load_model(["v3_tomato_price_pipeline.joblib"])
    model_y, path_y = _load_model(["yield_prediction_model.joblib"])
    df_market = _get_recent_market_data()
    latest_price = _get_latest_market_price()

    candidates = [95, 105, 115]
    evaluation_results = []

    # 폴백 발생 여부를 추적하기 위한 변수
    is_fallback = False
    fallback_reason = None
    fallback_message = "재배 및 환경데이터를 기반으로 계산되었습니다."

    # 2. 루프 시작
    for days in candidates:
        target_date = datetime.combine(p_date + timedelta(days=days), datetime.min.time())

        # --- [A] 가격 예측 ---
        final_price = 3500.0
        if model_p and not df_market.empty:
            try:
                price_s = df_market["price_per_kg"]
                temp_s = df_market["AVG_TEMP"]
                input_p = pd.DataFrame([{
                    "MA_30D": price_s.tail(30).mean(),
                    "MA_90D": price_s.tail(90).mean(),
                    "YOY_RATIO": float(price_s.iloc[-1]) / price_s.iloc[-365] if len(price_s) >= 365 else 1.0,
                    "GDD_30D": (temp_s.tail(30) - 10).clip(lower=0).sum(),
                    "TEMP_MA_30D": temp_s.tail(30).mean(),
                    "WEEK_SIN": np.sin(2 * np.pi * target_date.isocalendar()[1] / 52),
                    "WEEK_COS": np.cos(2 * np.pi * target_date.isocalendar()[1] / 52),
                    "MONTH": target_date.month
                }])
                final_price = float(np.expm1(model_p.predict(input_p)[0]))
            except:
                final_price = latest_price
        if final_price <= 100: final_price = latest_price

        # --- [B] 생산량 예측 ---
        final_yield_total = 0.0
        current_yield_reason = None

        if model_y and cult_id:
            try:
                X_y, feature_error = _build_yield_feature_row(int(cult_id))
                if X_y is not None:
                    X_y["snapshot_day"] = _nearest_snapshot_day(days)
                    final_yield_total = max(0.0, float(np.expm1(model_y.predict(X_y)[0])))
                else:
                    current_yield_reason = "feature_unavailable"
                    fallback_message = feature_error
            except:
                current_yield_reason = "predict_error"

        # 모델 예측 실패 시 폴백 적용
        if final_yield_total <= 0:
            final_yield_total = area * 30.0
            is_fallback = True
            if not current_yield_reason: current_yield_reason = "model_unavailable"
            fallback_reason = current_yield_reason

        evaluation_results.append({
            "days": days,
            "date": target_date,
            "price": final_price,
            "yield": final_yield_total,
            "expected_sales": final_yield_total * final_price,
            "reason": current_yield_reason
        })

    # 3. 최적 시점 결정
    best_option = max(evaluation_results, key=lambda x: x['expected_sales'])

    # 4. 결과 반환
    days_to_go = max(0, best_option["days"] - days_passed)

    final_is_fallback = (best_option["reason"] is not None)

    result = {
        "expected_harvest_date": best_option["date"].date().isoformat(),
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
        "price_day_95": next((r["price"] for r in evaluation_results if r["days"] == 95), 0),
        "price_day_105": next((r["price"] for r in evaluation_results if r["days"] == 105), 0),
        "price_day_115": next((r["price"] for r in evaluation_results if r["days"] == 115), 0),
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