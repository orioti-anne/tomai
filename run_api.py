import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()
import oracledb
oracledb.init_oracle_client(lib_dir="/opt/oracle")
print("[Oracle] thick mode 초기화 완료")
from api_server import app
from waitress import serve
print("[Server] 시작")
serve(app, host="0.0.0.0", port=8000)
