from smartfarm.ml.model_registry import load_price_model
from smartfarm.ml.features.price_features import (
    build_price_features,
    fill_missing_price_features,
    to_dataframe,
    DEFAULT_PRICE_FEATURES,
)


def predict_price(target_date, market_meta: dict, price_history: dict, weather_summary: dict) -> dict:
    bundle = load_price_model()
    model = bundle["model"]
    feature_order = bundle.get("features") or DEFAULT_PRICE_FEATURES

    features = build_price_features(
        target_date=target_date,
        market_meta=market_meta,
        price_history=price_history,
        weather_summary=weather_summary,
    )
    features = fill_missing_price_features(features)

    X = to_dataframe(features, feature_order=feature_order)
    pred = model.predict(X)[0]

    return {
        "predicted_price_per_kg": max(float(pred), 0.0),
        "feature_source": "price_model",
        "used_features": features,
    }