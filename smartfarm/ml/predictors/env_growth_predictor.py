from smartfarm.ml.model_registry import load_env_growth_model
from smartfarm.ml.features.env_features import build_env_growth_features, to_dataframe


def predict_growth_from_env(env_summary: dict, latest_growth: dict | None, dap: int) -> dict:
    bundle = load_env_growth_model()
    model = bundle["model"]
    feature_order = bundle.get("features")

    features = build_env_growth_features(
        env_summary=env_summary,
        latest_growth=latest_growth,
        dap=dap,
    )
    X = to_dataframe(features, feature_order=feature_order)

    pred = model.predict(X)[0]

    return {
        "predicted_rgr_height": float(pred),
        "feature_source": "env_model",
        "used_features": features,
    }