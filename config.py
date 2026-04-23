import os
from dotenv import load_dotenv

load_dotenv()

WEATHER_SERVICE_KEY = os.getenv("WEATHER_SERVICE_KEY")

SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "postgresql://macmini@localhost:5432/tomaidb")
SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev")

ENV_INGEST_URL = "http://127.0.0.1:5000/api/environment/ingest"
VIRTUAL_SENSOR_ENABLED_BY_DEFAULT = False
