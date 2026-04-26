import os
from dotenv import load_dotenv
load_dotenv(override=False)

from smartfarm import create_app
from waitress import serve

app = create_app(enable_scheduler=True)

if __name__ == "__main__":
    print("[Web] 맥 미니 웹서버 시작")
    serve(app, host="0.0.0.0", port=5000)

