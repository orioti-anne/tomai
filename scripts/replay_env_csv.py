import csv
import time
import requests

URL = "http://127.0.0.1:5000/api/environment/ingest"
CSV_PATH = "data/sample_environment.csv"


with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        payload = {
            "cult_id": row.get("cult_id"),
            "measure_time": row.get("measure_time"),
            "out_temp": row.get("out_temp"),
            "out_wind_direction": row.get("out_wind_direction"),
            "out_wind_speed": row.get("out_wind_speed"),
            "out_solar_rad": row.get("out_solar_rad"),
            "out_acc_solar_rad": row.get("out_acc_solar_rad"),
            "rain_detection": row.get("rain_detection"),
            "in_temp": row.get("in_temp"),
            "in_humidity": row.get("in_humidity"),
            "in_co2": row.get("in_co2"),
            "soil_temp": row.get("soil_temp"),
        }

        response = requests.post(URL, json=payload, timeout=10)
        print(response.status_code, response.text)
        time.sleep(3)