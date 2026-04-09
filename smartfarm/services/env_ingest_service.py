from datetime import datetime

from smartfarm import db
from smartfarm.models import Environment


def parse_datetime(value):
    if not value:
        raise ValueError("measure_time은 필수입니다.")

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise ValueError("measure_time 형식이 올바르지 않습니다. 예: 2026-04-01 10:00:00")


def to_float(value):
    if value in (None, "", "null"):
        return None
    return float(value)


def to_int(value):
    if value in (None, "", "null"):
        return None
    return int(value)


def ingest_environment_payload(data):
    cult_id = data.get("cult_id")
    measure_time = parse_datetime(data.get("measure_time"))

    if not cult_id:
        raise ValueError("cult_id는 필수입니다.")

    env = Environment(
        cult_id=int(cult_id),
        measure_time=measure_time,
        out_temp=to_float(data.get("out_temp")),
        out_wind_direction=to_float(data.get("out_wind_direction")),
        out_wind_speed=to_float(data.get("out_wind_speed")),
        out_solar_rad=to_float(data.get("out_solar_rad")),
        out_acc_solar_rad=to_float(data.get("out_acc_solar_rad")),
        rain_detection=to_int(data.get("rain_detection")),
        in_temp=to_float(data.get("in_temp")),
        in_humidity=to_float(data.get("in_humidity")),
        in_co2=to_float(data.get("in_co2")),
        soil_temp=to_float(data.get("soil_temp")),
    )

    db.session.add(env)
    db.session.commit()

    return {
        "env_id": env.env_id,
        "cult_id": env.cult_id,
        "measure_time": env.measure_time.strftime("%Y-%m-%d %H:%M:%S")
    }