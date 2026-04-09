import sys
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import create_app


def main():
    csv_path = os.path.join(project_root, "data", "환경_2022_원본 - 시트3.csv")

    print("1. CSV 읽기 시작", flush=True)
    df = pd.read_csv(csv_path, encoding="cp949")
    print("2. CSV 읽기 완료:", len(df), flush=True)

    app = create_app()
    print("3. app 생성 완료", flush=True)

    with app.app_context():
        print("4. ingest import 시작", flush=True)
        from smartfarm.services.env_pipeline_service import ingest_environment_data
        print("5. ingest import 완료", flush=True)

        print("6. ingest 실행 시작", flush=True)
        result = ingest_environment_data(df)
        print("7. ingest 완료", flush=True)
        print(result, flush=True)


if __name__ == "__main__":
    main()