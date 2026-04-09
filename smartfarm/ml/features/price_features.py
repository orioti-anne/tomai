from datetime import datetime
import pandas as pd


DEFAULT_PRICE_FEATURES = [
    "GRADE_SCORE",
    "UNIT_KG",
    "MONTH",
    "WEEK",
    "PREV_PER_KG_1D",
    "PRICE_MA_3D",
    "PRICE_DIFF",
    "AVG_TEMP",
    "SUNSHINE",
    "TEMP_LAG7",
]


def build_price_features(
    target_date: datetime,
    market_meta: dict,
    price_history: dict,
    weather_summary: dict,
) -> dict:
    month = target_date.month
    week = int(target_date.isocalendar().week)

    return {
        "GRADE_SCORE": market_meta.get("grade_score", 3),
        "UNIT_KG": market_meta.get("unit_kg", 5),
        "MONTH": month,
        "WEEK": week,
        "PREV_PER_KG_1D": price_history.get("prev_per_kg_1d"),
        "PRICE_MA_3D": price_history.get("price_ma_3d"),
        "PRICE_DIFF": price_history.get("price_diff"),
        "AVG_TEMP": weather_summary.get("avg_temp"),
        "SUNSHINE": weather_summary.get("sunshine"),
        "TEMP_LAG7": weather_summary.get("temp_lag7"),
    }


def fill_missing_price_features(feature_dict: dict) -> dict:
    defaults = {
        "PREV_PER_KG_1D": 3000.0,
        "PRICE_MA_3D": 3000.0,
        "PRICE_DIFF": 0.0,
        "AVG_TEMP": 20.0,
        "SUNSHINE": 6.0,
        "TEMP_LAG7": 20.0,
    }

    result = feature_dict.copy()
    for key, default_value in defaults.items():
        if result.get(key) is None:
            result[key] = default_value

    return result


def to_dataframe(feature_dict: dict, feature_order: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame([feature_dict])

    if feature_order:
        for col in feature_order:
            if col not in df.columns:
                df[col] = None
        df = df[feature_order]

    return df