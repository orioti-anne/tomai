from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]   # project root
MODEL_DIR = Path(__file__).resolve().parent / "models"

ENV_GROWTH_MODEL_PATH = MODEL_DIR / "env_growth_model.joblib"
PRODUCTION_MODEL_PATH = MODEL_DIR / "prod_growth_model.joblib"
PRICE_MODEL_PATH = MODEL_DIR / "v3_tomato_price_pipeline.joblib"