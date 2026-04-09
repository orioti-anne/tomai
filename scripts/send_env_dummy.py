import random
from datetime import datetime, timedelta

import requests

URL = "http://127.0.0.1:5000/api/environment/ingest"
CULT_ID = 1


def get_target_measure_time(now=None):
    now = now or datetime.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    return current_hour - timedelta(hours=1)


def build_payload(cult_id, measure_time):
    return {
        "cult_id": cult_id,
        "measure_time": measure_time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_temp": round(random.uniform(10, 25), 2),
        "out_wind_direction": round(random.uniform(0, 360), 2),
        "out_wind_speed": round(random.uniform(0, 5), 2),
        "out_solar_rad": round(random.uniform(0, 900), 2),
        "out_acc_solar_rad": round(random.uniform(0, 5000), 2),
        "rain_detection": random.choice([0, 0, 0, 1]),
        "in_temp": round(random.uniform(18, 30), 2),
        "in_humidity": round(random.uniform(50, 90), 2),
        "in_co2": round(random.uniform(350, 1000), 2),
        "soil_temp": round(random.uniform(15, 25), 2),
    }


def main():
    target_time = get_target_measure_time()
    payload = build_payload(CULT_ID, target_time)

    response = requests.post(URL, json=payload, timeout=10)
    print("target_time =", target_time.strftime("%Y-%m-%d %H:%M:%S"))
    print(response.status_code, response.text)


if __name__ == "__main__":
    main()