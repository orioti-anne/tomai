from smartfarm.ml.model_registry import load_production_model
from smartfarm.ml.features.growth_features import (
    build_production_features,
    fill_missing_production_features,
    to_dataframe,
    DEFAULT_PRODUCTION_FEATURES,
)


def predict_production(growth_data: dict, dap: int) -> dict:
    bundle = load_production_model()
    model = bundle["model"]
    feature_order = bundle.get("features") or DEFAULT_PRODUCTION_FEATURES

    features = build_production_features(growth_data=growth_data, dap=dap)
    features = fill_missing_production_features(features)

    X = to_dataframe(features, feature_order=feature_order)
    pred = model.predict(X)[0]

    return {
        "predicted_quantity": max(float(pred), 0.0),
        "feature_source": "growth_model",
        "used_features": features,
    }