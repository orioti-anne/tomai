from __future__ import annotations

import math
from typing import Any, Dict, Optional

import pandas as pd


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def calculate_vpd(temp: Optional[float], humid: Optional[float]) -> Optional[float]:
    """
    temp: ℃
    humid: %
    return: kPa
    """
    temp = safe_float(temp)
    humid = safe_float(humid)

    if temp is None or humid is None:
        return None
    if humid <= 0:
        return None

    es = 0.61078 * math.exp((17.27 * temp) / (temp + 237.3))
    ea = es * (humid / 100.0)
    return es - ea


def build_env_growth_features(
    env_summary: Dict[str, Any],
    latest_growth: Optional[Dict[str, Any]],
    dap: int,
) -> Dict[str, Any]:

    avg_temp = safe_float(env_summary.get("avg_temp"))
    avg_humid = safe_float(env_summary.get("avg_humid"))
    avg_co2 = safe_float(env_summary.get("avg_co2"))
    daily_solar = safe_float(env_summary.get("daily_solar"))
    high_temp_hours = safe_float(env_summary.get("high_temp_hours"))

    vpd = calculate_vpd(avg_temp, avg_humid)

    prev_height = None
    leaf_count = None

    if latest_growth:
        prev_height = safe_float(latest_growth.get("plant_height"))
        leaf_count = safe_float(latest_growth.get("leaf_count"))

    return {
        "PERIOD_GDD": max((avg_temp or 0) - 10, 0),
        "PERIOD_VPD": vpd,
        "PERIOD_SOLAR_ACC": daily_solar,
        "VPD_SOLAR_INTERACT": (vpd or 0) * (daily_solar or 0),
        "HIGH_TEMP_SUM": high_temp_hours,
        "PERIOD_CO2_AVG": avg_co2,
        "PREV_HEIGHT": prev_height,
        "LEAF_COUNT": leaf_count,
        "DAP": safe_float(dap, 0.0),
    }


def to_dataframe(feature_dict: Dict[str, Any], feature_order: Optional[list[str]] = None) -> pd.DataFrame:
    df = pd.DataFrame([feature_dict])

    if feature_order:
        for col in feature_order:
            if col not in df.columns:
                df[col] = None
        df = df[feature_order]

    return df



def build_env_recommendation_features(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    실시간/최근 센서 입력을 추천 모델용 피처로 변환
    """
    avg_temp = safe_float(sensor_data.get("avg_temp", sensor_data.get("temp")))
    avg_humid = safe_float(sensor_data.get("avg_humid", sensor_data.get("humid")))
    avg_co2 = safe_float(sensor_data.get("avg_co2", sensor_data.get("co2")))
    solar_acc = safe_float(sensor_data.get("solar_acc", sensor_data.get("solar")))
    soil_temp = safe_float(sensor_data.get("soil_temp"))
    dap = safe_float(sensor_data.get("dap"), 0.0)
    high_temp_hours = safe_float(sensor_data.get("high_temp_hours"), 0.0)

    if high_temp_hours == 0 and avg_temp is not None and avg_temp >= 30:
        high_temp_hours = 1.0

    vpd = calculate_vpd(avg_temp, avg_humid)

    return {
        "AVG_TEMP": avg_temp,
        "AVG_HUMID": avg_humid,
        "AVG_CO2": avg_co2,
        "SOLAR_ACC": solar_acc,
        "SOIL_TEMP": soil_temp,
        "HIGH_TEMP_HOURS": high_temp_hours,
        "VPD": vpd,
        "DAP": dap,
    }


def compare_with_optimal_profile(
    current_features: Dict[str, Any],
    optimal_profile: Dict[str, Dict[str, float]],
) -> list[dict]:
    """
    현재값과 최적 프로파일 비교
    """
    recommendations: list[dict] = []

    label_map = {
        "AVG_TEMP": ("온도", "℃"),
        "AVG_HUMID": ("습도", "%"),
        "AVG_CO2": ("CO₂", "ppm"),
        "SOLAR_ACC": ("일사량", "W/m²"),
        "SOIL_TEMP": ("토양온도", "℃"),
        "VPD": ("VPD", "kPa"),
    }

    for key, (label, unit) in label_map.items():
        if key not in optimal_profile:
            continue

        current = safe_float(current_features.get(key))
        if current is None:
            continue

        target_info = optimal_profile[key]
        low = safe_float(target_info.get("min"))
        high = safe_float(target_info.get("max"))
        best = safe_float(target_info.get("best"))

        if low is None or high is None or best is None:
            continue

        if current < low:
            recommendations.append({
                "factor_key": key,
                "factor": label,
                "status": "low",
                "current": round(current, 2),
                "recommended": round(best, 2),
                "range_min": round(low, 2),
                "range_max": round(high, 2),
                "unit": unit,
                "message": f"{label}이 낮습니다. {best}{unit} 수준까지 올리는 것이 좋습니다.",
            })
        elif current > high:
            recommendations.append({
                "factor_key": key,
                "factor": label,
                "status": "high",
                "current": round(current, 2),
                "recommended": round(best, 2),
                "range_min": round(low, 2),
                "range_max": round(high, 2),
                "unit": unit,
                "message": f"{label}이 높습니다. {best}{unit} 수준으로 낮추는 것이 좋습니다.",
            })

    return recommendations