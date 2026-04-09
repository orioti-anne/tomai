import joblib

from smartfarm.ml.config import (
    ENV_GROWTH_MODEL_PATH,
    PRODUCTION_MODEL_PATH,
    PRICE_MODEL_PATH,
)


def _normalize_bundle(bundle):
    """
    저장 포맷이 아래 둘 중 어느 쪽이든 맞춰서 반환:
    1) model 객체만 저장한 경우
    2) {'model': ..., 'features': [...], ...} 형태로 저장한 경우
    """
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle

    return {
        "model": bundle,
        "features": None,
        "metrics": None,
    }


def load_env_growth_model():
    bundle = joblib.load(ENV_GROWTH_MODEL_PATH)
    return _normalize_bundle(bundle)


def load_production_model():
    bundle = joblib.load(PRODUCTION_MODEL_PATH)
    return _normalize_bundle(bundle)


def load_price_model():
    bundle = joblib.load(PRICE_MODEL_PATH)
    return _normalize_bundle(bundle)