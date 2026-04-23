import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text

# 1. 경로 설정 (현재 실행 중인 파일의 위치를 기준으로 프로젝트 루트 탐색)
current_file_path = os.path.abspath(__file__) # 현재 파일 경로
services_dir = os.path.dirname(current_file_path) # services 폴더
ml_dir = os.path.dirname(services_dir) # ml 폴더
project_root = os.path.dirname(ml_dir) # AI_project1 폴더

if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app

def predict_future():
    app = create_app(enable_scheduler=False)

    with app.app_context():
        # PostgreSQL LIMIT 방식
        query = text("""
            SELECT p.price_date, p.price_per_kg, w.avg_temp, w.sunshine, w.rain
            FROM kamis_tomato_price p
            LEFT JOIN weather_index w ON p.price_date = w.w_date
            WHERE p.item_name = '완숙토마토'
            ORDER BY p.price_date DESC
            LIMIT 40
        """)

        try:
            # 데이터 로드 및 정렬
            df = pd.read_sql(query, db.engine)
            df.columns = [c.lower() for c in df.columns]
            df = df.sort_values("price_date")

            if len(df) < 30:
                print(f"⚠️ 데이터 부족: 현재 {len(df)}건의 데이터만 확보되었습니다. 최소 30일치가 필요합니다.")
                return

            # [핵심 수정] 모델 파일 경로 동적 탐색
            # AI_project1/models/v3_tomato_price_pipeline.joblib 위치를 찾아갑니다.
            model_path = os.path.join(project_root,"ml", "models", "v3_tomato_price_pipeline.joblib")

            if not os.path.exists(model_path):
                print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
                print("학습 스크립트(train_price_model_v3.py)를 먼저 실행하여 모델을 생성해주세요.")
                return

            pipeline = joblib.load(model_path)

            # 피처 생성 (V3 학습 로직과 동일)
            latest = df.iloc[-1].copy()
            prev_1d = df["price_per_kg"].iloc[-1]
            ma_7d = df["price_per_kg"].tail(7).mean()
            ma_30d = df["price_per_kg"].tail(30).mean()
            price_vol_7d = df["price_per_kg"].tail(7).std()

            rain_sum_7d = df["rain"].tail(7).sum()
            sun_avg_10d = df["sunshine"].tail(10).mean()
            temp_lag14 = df["avg_temp"].iloc[-15] if len(df) >= 15 else df["avg_temp"].mean()

            today = datetime.now()
            target_date = today + timedelta(days=7)

            # 입력 데이터 프레임 구성
            input_data = pd.DataFrame([{
                "PREV_1D": prev_1d,
                "MA_7D": ma_7d,
                "MA_30D": ma_30d,
                "PRICE_VOL_7D": price_vol_7d,
                "avg_temp": latest["avg_temp"],
                "TEMP_LAG14": temp_lag14,
                "RAIN_SUM_7D": rain_sum_7d,
                "SUN_AVG_10D": sun_avg_10d,
                "WEEK_SIN": np.sin(2 * np.pi * today.isocalendar()[1] / 52),
                "WEEK_COS": np.cos(2 * np.pi * today.isocalendar()[1] / 52),
                "MONTH": today.month,
                "YEAR": today.year
            }])

            # 예측 수행
            prediction = pipeline.predict(input_data)[0]

            print(f"\n--- 🍅 완숙토마토 7일 뒤 시세 예측 결과 ---")
            print(f"📅 예측 기준일: {today.strftime('%Y-%m-%d')}")
            print(f"🔮 예측 목표일: {target_date.strftime('%Y-%m-%d')}")
            print(f"💰 현재 가격: {int(prev_1d):,}원 / 1kg")
            print(f"🚀 예측 가격: {int(prediction):,}원 / 1kg")
            print(f"📈 예상 변동: {int(prediction - prev_1d):+}원")
            print(f"----------------------------------------\n")

        except Exception as e:
            print(f"❌ 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    predict_future()