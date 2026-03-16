import os
import oracledb
import platform
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_SERVICE = os.getenv("DB_SERVICE")
ORACLE_PATH = os.getenv("ORACLE_CLIENT_PATH")

try:
    if platform.system() == "Darwin":
        oracledb.init_oracle_client(lib_dir=ORACLE_PATH)
    elif platform.system() == "Windows":
        oracledb.init_oracle_client(lib_dir=r"C:\oraclexe\instantclient_19_25")
except Exception as e:
    print(f"⚠️ Oracle 설정 알림: {e}")

SQLALCHEMY_DATABASE_URI = f"oracle+oracledb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_SERVICE}"
SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev")