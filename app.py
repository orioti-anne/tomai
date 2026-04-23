import os
from dotenv import load_dotenv
load_dotenv(override=False)

import oracledb
import platform
try:
    if platform.system() == "Windows":
        oracledb.init_oracle_client(lib_dir=r"C:\oraclexe\instantclient_19_25")
    else:
        oracledb.init_oracle_client(lib_dir=os.getenv("ORACLE_CLIENT_PATH"))
    print("[Oracle] thick mode 초기화 완료")
except Exception as e:
    print(f"[Oracle] 설정 알림: {e}")

from smartfarm import create_app
app = create_app()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
