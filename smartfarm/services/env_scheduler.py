import os
import random
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import text

from smartfarm import db
from smartfarm.models import Cultivations
from .data_collector_service import PriceCollector

import requests

load_dotenv()

GCP_IP = os.getenv("GCP_IP")
if not GCP_IP:
    raise ValueError("환경 변수 GCP_IP가 설정되지 않았습니다. .env 파일을 확인하세요.")

GCP_ENDPOINT = f"http://{GCP_IP}:5000/api/receive-prediction"

def send_result_to_gcp(category: str, value: float, target_date: str):
    payload = {"type": category, "value": value, "target_date": target_date}
    try:
        response = requests.post(GCP_ENDPOINT, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"[GCP 전송 성공] {category}: {value}")
        else:
            print(f"[GCP 전송 응답 오류] 상태코드: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"[GCP 전송 실패] 네트워크 연결 확인 필요: {e}")
    except Exception as e:
        print(f"[GCP 전송 실패] {type(e).__name__}: {e}")


scheduler = BackgroundScheduler(timezone="Asia/Seoul")

# 기상청 ASOS 일자료 API
KMA_ASOS_URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"

# 토마토 주산지 관측소
LOCATIONS = {
    "부여": "236",
    "금산": "238",
    "창원": "155",
    "전주": "146",
    "진주": "192",
}


def _to_float(value, default=0.0):
    if value in (None, "", "-"):
        return default
    try:
        return float(value)
    except Exception:
        return default


def get_target_measure_time(now=None):
    now = now or datetime.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    return current_hour - timedelta(hours=1)


def _get_solar_base_by_hour(hour: int) -> int:
    if 0 <= hour < 6:
        return 0
    elif 6 <= hour < 10:
        return 200
    elif 10 <= hour < 16:
        return 700
    elif 16 <= hour < 20:
        return 150
    else:
        return 0


def _generate_hourly_solar_rad(hour: int) -> float:
    solar_base = _get_solar_base_by_hour(hour)
    return round(max(0, random.uniform(solar_base * 0.8, solar_base * 1.2)), 1)


def _calculate_daily_acc_solar_rad(current_hour: int) -> float:
    acc_value = 0.0

    for h in range(current_hour + 1):
        hourly_solar = _generate_hourly_solar_rad(h)
        acc_value += hourly_solar * 0.36

    return round(acc_value, 1)


def build_payload(cult_id, measure_time):
    hour = measure_time.hour

    if 0 <= hour < 6:
        out_temp_base = 11
        in_temp_base = 18
    elif 6 <= hour < 10:
        out_temp_base = 15
        in_temp_base = 21
    elif 10 <= hour < 16:
        out_temp_base = 21
        in_temp_base = 26
    elif 16 <= hour < 20:
        out_temp_base = 17
        in_temp_base = 22
    else:
        out_temp_base = 13
        in_temp_base = 19

    out_solar_rad = _generate_hourly_solar_rad(hour)
    out_acc_solar_rad = _calculate_daily_acc_solar_rad(hour)

    return {
        "cult_id": cult_id,
        "measure_time": measure_time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_temp": round(random.uniform(out_temp_base - 2, out_temp_base + 2), 1),
        "out_wind_direction": random.randint(0, 360),
        "out_wind_speed": round(random.uniform(0.0, 4.0), 1),
        "out_solar_rad": out_solar_rad,
        "out_acc_solar_rad": out_acc_solar_rad,
        "rain_detection": 1 if random.random() < 0.08 else 0,
        "in_temp": round(random.uniform(in_temp_base - 2, in_temp_base + 2), 1),
        "in_humidity": round(random.uniform(55.0, 85.0), 1),
        "in_co2": random.randint(350, 900),
        "soil_temp": round(random.uniform(16.0, 24.0), 1),
    }


def get_enabled_cult_ids():
    rows = Cultivations.query.filter(Cultivations.virtual_sensor_enabled == "Y").all()
    return [row.cult_id for row in rows]


def run_environment_hourly_job(app):
    with app.app_context():
        url = app.config.get("ENV_INGEST_URL", "http://127.0.0.1:5000/api/environment/ingest")
        cult_ids = get_enabled_cult_ids()
        target_time = get_target_measure_time()

        if not cult_ids:
            print("[ENV_SCHEDULER] 활성화된 cult_id가 없습니다.")
            return

        for cult_id in cult_ids:
            payload = build_payload(cult_id, target_time)
            try:
                response = requests.post(url, json=payload, timeout=10)
                print(
                    f"[ENV_SCHEDULER] 데이터 전송 성공: "
                    f"cult_id={cult_id}, status={response.status_code}"
                )
            except Exception as e:
                print(f"[ENV_SCHEDULER][ERROR] 전송 실패: cult_id={cult_id}, error={e}")


def run_price_collect_job(app):
    with app.app_context():
        print(f"[{datetime.now()}] --- 시세 수집 스케줄러 가동 ---")
        try:
            result = PriceCollector.collect_tomato_price()
            if result.get("success"):
                print(f"[{datetime.now()}] 성공: {result.get('details')}")

                price_val = result.get('price', 0)
                target_date = datetime.now().strftime('%Y-%m-%d')
                send_result_to_gcp("price", price_val, target_date)

            else:
                print(f"[{datetime.now()}] 실패: {result.get('message')}")
        except Exception as e:
            print(f"[{datetime.now()}] 시세 수집 예외: {e}")


def run_weather_collect_job(app):
    """
    기상청 ASOS 일자료를 하루 1번 가져와
    5개 지점 평균값을 WEATHER_INDEX에 upsert
    """
    with app.app_context():
        service_key = app.config.get("WEATHER_SERVICE_KEY")
        if not service_key:
            print("[WEATHER_SCHEDULER] WEATHER_SERVICE_KEY 설정이 없습니다.")
            return

        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        weather_date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"[WEATHER_SCHEDULER] 수집 시작: target_date={target_date}")

        all_weather_data = []

        for loc_name, loc_code in LOCATIONS.items():
            params = {
                "ServiceKey": service_key,
                "numOfRows": "999",
                "pageNo": "1",
                "dataType": "JSON",
                "dataCd": "ASOS",
                "dateCd": "DAY",
                "startDt": target_date,
                "endDt": target_date,
                "stnIds": loc_code,
            }

            try:
                res = requests.get(KMA_ASOS_URL, params=params, timeout=20)

                if res.status_code != 200:
                    print(f"[WEATHER_SCHEDULER][ERROR] {loc_name}({loc_code}) status={res.status_code}")
                    print(f"[WEATHER_SCHEDULER][ERROR] response={res.text[:500]}")
                    continue

                data = res.json()

                items = (
                    data.get("response", {})
                    .get("body", {})
                    .get("items", {})
                    .get("item", [])
                )

                if not items:
                    print(f"[WEATHER_SCHEDULER] {loc_name}({loc_code}) 데이터 없음")
                    continue

                for item in items:
                    all_weather_data.append({
                        "W_DATE": item.get("tm"),
                        "AVG_TEMP": _to_float(item.get("avgTa"), 0.0),
                        "SUNSHINE": _to_float(item.get("sumSsHr"), 0.0),
                        "RAIN": _to_float(item.get("sumRn"), 0.0),
                        "HUMID": _to_float(item.get("avgRhm"), 0.0),
                    })

                print(f"[WEATHER_SCHEDULER] {loc_name}({loc_code}) 완료")

            except Exception as e:
                print(f"[WEATHER_SCHEDULER][ERROR] {loc_name}({loc_code}) 실패: {e}")

            time.sleep(0.3)

        if not all_weather_data:
            print("[WEATHER_SCHEDULER] 수집된 데이터가 없습니다.")
            return

        df = pd.DataFrame(all_weather_data)
        df["W_DATE"] = pd.to_datetime(df["W_DATE"], errors="coerce")
        df = df.dropna(subset=["W_DATE"])

        if df.empty:
            print("[WEATHER_SCHEDULER] 유효한 날짜 데이터가 없습니다.")
            return

        national_df = df.groupby("W_DATE", as_index=False)[["AVG_TEMP", "SUNSHINE", "RAIN", "HUMID"]].mean()

        if national_df.empty:
            print("[WEATHER_SCHEDULER] 평균 계산 결과가 없습니다.")
            return

        row = national_df.iloc[0]
        avg_temp = round(_to_float(row["AVG_TEMP"], 0.0), 2)
        sunshine = round(_to_float(row["SUNSHINE"], 0.0), 2)
        rain = round(_to_float(row["RAIN"], 0.0), 2)
        humid = round(_to_float(row["HUMID"], 0.0), 2)

        print(
            f"[WEATHER_SCHEDULER] 평균 완료: "
            f"date={weather_date_str}, temp={avg_temp}, sun={sunshine}, rain={rain}, humid={humid}"
        )

        try:
            merge_sql = text("""
                MERGE INTO WEATHER_INDEX T
                USING (
                    SELECT
                        TO_DATE(:w_date, 'YYYY-MM-DD') AS W_DATE,
                        :avg_temp AS AVG_TEMP,
                        :sunshine AS SUNSHINE,
                        :rain AS RAIN,
                        :humid AS HUMID
                    FROM DUAL
                ) S
                ON (T.W_DATE = S.W_DATE)
                WHEN MATCHED THEN
                    UPDATE SET
                        T.AVG_TEMP = S.AVG_TEMP,
                        T.SUNSHINE = S.SUNSHINE,
                        T.RAIN = S.RAIN,
                        T.HUMID = S.HUMID
                WHEN NOT MATCHED THEN
                    INSERT (W_DATE, AVG_TEMP, SUNSHINE, RAIN, HUMID)
                    VALUES (S.W_DATE, S.AVG_TEMP, S.SUNSHINE, S.RAIN, S.HUMID)
            """)

            db.session.execute(
                merge_sql,
                {
                    "w_date": weather_date_str,
                    "avg_temp": avg_temp,
                    "sunshine": sunshine,
                    "rain": rain,
                    "humid": humid,
                },
            )
            db.session.commit()

            print(f"[WEATHER_SCHEDULER] WEATHER_INDEX 저장 완료: {weather_date_str}")

            send_result_to_gcp("weather", avg_temp, weather_date_str)

        except Exception as e:
            db.session.rollback()
            print(f"[WEATHER_SCHEDULER][ERROR] DB 저장 실패: {e}")


def init_scheduler(app):
    use_reloader = app.debug or app.config.get("DEBUG", False)

    if use_reloader and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return

    if scheduler.running:
        return

    scheduler.add_job(
        func=run_environment_hourly_job,
        trigger="cron",
        minute=0,
        id="environment_hourly_job",
        replace_existing=True,
        misfire_grace_time=900,
        coalesce=True,
        max_instances=1,
        args=[app],
    )

    scheduler.add_job(
        func=run_price_collect_job,
        trigger="cron",
        hour=10,
        minute=15,
        id="daily_price_collect_job",
        replace_existing=True,
        misfire_grace_time=900,
        coalesce=True,
        max_instances=1,
        args=[app],
    )

    scheduler.add_job(
        func=run_weather_collect_job,
        trigger="cron",
        hour=10,
        minute=10,
        id="daily_weather_collect_job",
        replace_existing=True,
        misfire_grace_time=900,
        coalesce=True,
        max_instances=1,
        args=[app],
    )

    add_sync_job(scheduler, app)
    scheduler.start()
    print("[SCHEDULER] 통합 스케줄러가 시작되었습니다. (센서: 매시간, 시세/기상: 오전 1회)")

def add_sync_job(scheduler, app):
    from smartfarm.services.cloud_sync_service import run_full_sync
    scheduler.add_job(
        run_full_sync,
        'interval',
        hours=1,
        args=[app],
        id='cloud_sync',
        replace_existing=True
    )
    print("[SCHEDULER] 클라우드 동기화 스케줄러 추가 완료")
