from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

import joblib
import pandas as pd

from smartfarm.ml.features.env_features import (
    build_env_recommendation_features,
    compare_with_optimal_profile,
)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

@lru_cache(maxsize=None)
def _load_joblib(filename: str):
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    print(f"[model] loading: {path}")
    return joblib.load(path)


def _grade_growth_score(score: Optional[float]) -> str:
    if score is None:
        return "알수없음"
    if score >= 0.80:
        return "좋음"
    if score >= 0.60:
        return "보통"
    return "나쁨"


def _build_feature_impacts(
    current_features: Dict[str, Any],
    optimal_profile: Dict[str, Dict[str, float]],
) -> list[dict]:
    """
    현재값이 최적 범위에서 얼마나 벗어났는지 설명용 영향도 구성
    """
    impacts: list[dict] = []

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

        current = current_features.get(key)
        if current is None:
            continue

        target = optimal_profile[key].get("best")
        low = optimal_profile[key].get("min")
        high = optimal_profile[key].get("max")

        if target is None or low is None or high is None:
            continue

        gap = abs(float(current) - float(target))

        direction = "optimal"
        if current < low:
            direction = "low"
        elif current > high:
            direction = "high"

        impacts.append({
            "factor_key": key,
            "factor": label,
            "current": round(float(current), 2),
            "optimal": round(float(target), 2),
            "range_min": round(float(low), 2),
            "range_max": round(float(high), 2),
            "gap": round(gap, 2),
            "direction": direction,
            "unit": unit,
        })

    impacts.sort(key=lambda x: x["gap"], reverse=True)
    return impacts


def recommend_environment(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    model_bundle = _load_joblib("env_recommendation_model.joblib")
    optimal_profile = _load_joblib("env_optimal_profile.joblib")

    if model_bundle is None:
        raise FileNotFoundError("env_recommendation_model.joblib 파일이 없습니다.")

    if optimal_profile is None:
        raise FileNotFoundError("env_optimal_profile.joblib 파일이 없습니다.")

    model = model_bundle["model"]
    feature_order = model_bundle.get("features") or []

    current_features = build_env_recommendation_features(sensor_data)

    X = pd.DataFrame([current_features])
    for col in feature_order:
        if col not in X.columns:
            X[col] = None
    X = X[feature_order]

    predicted_growth_score = float(model.predict(X)[0])
    environment_grade = _grade_growth_score(predicted_growth_score)

    recommendations = compare_with_optimal_profile(current_features, optimal_profile)
    impacts = _build_feature_impacts(current_features, optimal_profile)

    top_issue = None
    if recommendations:
        top_issue = recommendations[0]["factor"]

    return {
        "predicted_growth_score": round(predicted_growth_score, 3),
        "environment_grade": environment_grade,
        "top_issue_factor": top_issue,
        "current_environment": current_features,
        "optimal_profile": optimal_profile,
        "feature_impacts": impacts,
        "recommendations": recommendations,
    }