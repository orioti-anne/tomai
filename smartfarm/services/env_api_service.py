from datetime import datetime, time

import numpy as np
import pandas as pd
from sqlalchemy import func

from smartfarm import db
from smartfarm.models import Cultivations, Environment, EnvCleaned, EnvSummary


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


def validate_cultivation(cult_id):
    cult = Cultivations.query.filter_by(cult_id=cult_id).first()
    if not cult:
        raise ValueError(f"유효하지 않은 cult_id입니다: {cult_id}")
    return cult


def build_environment_entity(data):
    cult_id = data.get("cult_id")
    if cult_id in (None, ""):
        raise ValueError("cult_id는 필수입니다.")

    cult_id = int(cult_id)
    validate_cultivation(cult_id)

    measure_time = parse_datetime(data.get("measure_time"))

    return Environment(
        cult_id=cult_id,
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


def process_chunk_final_v4(df):
    if df.empty:
        return df

    df = df.copy()

    df["measure_time"] = pd.to_datetime(df["measure_time"])
    df["measure_date"] = df["measure_time"].dt.normalize()
    df["measure_hour"] = df["measure_time"].dt.hour

    df["out_acc_solar_rad_status"] = 0
    df["in_temp_status"] = 0
    df["in_humidity_status"] = 0
    df["in_co2_status"] = 0

    df = df.sort_values(["cult_id", "measure_time"]).drop_duplicates(
        subset=["cult_id", "measure_time"], keep="first"
    )

    final_list = []

    for (cid, d_val), day_df in df.groupby(["cult_id", "measure_date"]):
        day_df = day_df.copy()

        env_ranges = {
            "in_temp": (-10, 55, "in_temp_status"),
            "in_humidity": (0, 100, "in_humidity_status"),
            "in_co2": (200, 3500, "in_co2_status"),
        }

        for col, (min_val, max_val, status_col) in env_ranges.items():
            outlier_mask = (day_df[col] < min_val) | (day_df[col] > max_val)
            if outlier_mask.any():
                day_df.loc[outlier_mask, status_col] = 9
                day_df.loc[outlier_mask, col] = np.nan

            missing_before = day_df[col].isna()
            if missing_before.any():
                day_df[col] = day_df[col].interpolate(method="linear", limit_direction="both")
                filled_mask = missing_before & day_df[col].notna()
                day_df.loc[filled_mask & (day_df[status_col] == 0), status_col] = 5

        night_mask = (day_df["measure_hour"] >= 19) | (day_df["measure_hour"] <= 6)
        day_df.loc[night_mask & (day_df["out_solar_rad"] > 20), "out_solar_rad"] = 0

        time_diff = day_df["measure_time"].diff().dt.total_seconds().fillna(0)
        day_df["calc_acc"] = (day_df["out_solar_rad"] * time_diff / 10000).cumsum()

        real_diff = day_df["out_acc_solar_rad"].max() - day_df["out_acc_solar_rad"].min()
        calc_max = day_df["calc_acc"].max()
        solar_mean = day_df["out_solar_rad"].mean()
        cv = (day_df["out_solar_rad"].std() / solar_mean) if solar_mean and solar_mean > 0 else 0
        base_tol = 0.15 if day_df["measure_hour"].between(10, 15).any() else 0.25
        adaptive_tolerance = base_tol + min(cv * 0.5, 0.15)

        is_mismatch = abs(real_diff - calc_max) > (calc_max * adaptive_tolerance) if calc_max else False
        is_stagnant = calc_max > 5 and real_diff < 5
        is_over_limit = (day_df["out_acc_solar_rad"] > 4000).any()

        if is_mismatch or is_stagnant or is_over_limit:
            day_df["out_acc_solar_rad"] = day_df["calc_acc"]
            day_df["out_acc_solar_rad_status"] = 6

        missing_acc_before = day_df["out_acc_solar_rad"].isna()
        if missing_acc_before.any():
            day_df.loc[missing_acc_before, "out_acc_solar_rad"] = day_df["calc_acc"]
            filled_acc_mask = missing_acc_before & day_df["out_acc_solar_rad"].notna()
            day_df.loc[
                filled_acc_mask & (day_df["out_acc_solar_rad_status"] == 0),
                "out_acc_solar_rad_status"
            ] = 5

        final_list.append(day_df)

    if not final_list:
        return pd.DataFrame()

    processed_df = pd.concat(final_list, ignore_index=True)

    target_columns = [
        "env_id", "cult_id", "measure_time", "out_temp",
        "out_wind_direction", "out_wind_speed", "out_solar_rad",
        "out_acc_solar_rad", "rain_detection", "in_temp",
        "in_humidity", "in_co2", "soil_temp",
        "out_acc_solar_rad_status", "in_temp_status", "in_humidity_status", "in_co2_status",
        "measure_date", "measure_hour"
    ]

    return processed_df[target_columns]


def rebuild_cleaned_for_day(cult_id, measure_date):
    start_dt = datetime.combine(measure_date, time.min)
    end_dt = datetime.combine(measure_date, time.max)

    rows = (
        Environment.query
        .filter(
            Environment.cult_id == cult_id,
            Environment.measure_time >= start_dt,
            Environment.measure_time <= end_dt,
        )
        .order_by(Environment.measure_time.asc())
        .all()
    )

    if not rows:
        return 0

    raw_df = pd.DataFrame([
        {
            "env_id": row.env_id,
            "cult_id": row.cult_id,
            "measure_time": row.measure_time,
            "out_temp": row.out_temp,
            "out_wind_direction": row.out_wind_direction,
            "out_wind_speed": row.out_wind_speed,
            "out_solar_rad": row.out_solar_rad,
            "out_acc_solar_rad": row.out_acc_solar_rad,
            "rain_detection": row.rain_detection,
            "in_temp": row.in_temp,
            "in_humidity": row.in_humidity,
            "in_co2": row.in_co2,
            "soil_temp": row.soil_temp,
        }
        for row in rows
    ])

    refined_df = process_chunk_final_v4(raw_df)
    if refined_df.empty:
        return 0

    EnvCleaned.query.filter(
        EnvCleaned.cult_id == cult_id,
        EnvCleaned.measure_date == measure_date
    ).delete(synchronize_session=False)

    cleaned_rows = []
    for _, row in refined_df.iterrows():
        measure_time = pd.to_datetime(row["measure_time"]).to_pydatetime()
        measure_date_value = pd.to_datetime(row["measure_date"]).date()

        cleaned_rows.append(
            EnvCleaned(
                env_id=int(row["env_id"]),
                cult_id=int(row["cult_id"]),
                measure_time=measure_time,
                out_temp=float(row["out_temp"]) if pd.notna(row["out_temp"]) else None,
                out_wind_direction=float(row["out_wind_direction"]) if pd.notna(row["out_wind_direction"]) else None,
                out_wind_speed=float(row["out_wind_speed"]) if pd.notna(row["out_wind_speed"]) else None,
                out_solar_rad=float(row["out_solar_rad"]) if pd.notna(row["out_solar_rad"]) else None,
                out_acc_solar_rad=float(row["out_acc_solar_rad"]) if pd.notna(row["out_acc_solar_rad"]) else None,
                rain_detection=int(row["rain_detection"]) if pd.notna(row["rain_detection"]) else None,
                in_temp=float(row["in_temp"]) if pd.notna(row["in_temp"]) else None,
                in_humidity=float(row["in_humidity"]) if pd.notna(row["in_humidity"]) else None,
                in_co2=float(row["in_co2"]) if pd.notna(row["in_co2"]) else None,
                soil_temp=float(row["soil_temp"]) if pd.notna(row["soil_temp"]) else None,
                out_acc_solar_rad_status=int(row["out_acc_solar_rad_status"]) if pd.notna(row["out_acc_solar_rad_status"]) else 0,
                in_temp_status=int(row["in_temp_status"]) if pd.notna(row["in_temp_status"]) else 0,
                in_humidity_status=int(row["in_humidity_status"]) if pd.notna(row["in_humidity_status"]) else 0,
                in_co2_status=int(row["in_co2_status"]) if pd.notna(row["in_co2_status"]) else 0,
                measure_date=measure_date_value,
                measure_hour=int(row["measure_hour"]) if pd.notna(row["measure_hour"]) else None,
            )
        )

    db.session.add_all(cleaned_rows)
    db.session.flush()

    return len(cleaned_rows)


def upsert_env_summary_from_cleaned(cult_id, measure_date):
    agg = (
        db.session.query(
            func.avg(EnvCleaned.out_temp),
            func.max(EnvCleaned.out_acc_solar_rad),
            func.max(EnvCleaned.rain_detection),
            func.avg(EnvCleaned.in_temp),
            func.avg(EnvCleaned.in_humidity),
            func.avg(EnvCleaned.in_co2),
            func.avg(EnvCleaned.soil_temp),
            func.avg(EnvCleaned.in_temp),
            func.max(EnvCleaned.out_acc_solar_rad),
        )
        .filter(
            EnvCleaned.cult_id == cult_id,
            EnvCleaned.measure_date == measure_date,
        )
        .one()
    )

    summary = (
        EnvSummary.query
        .filter(
            EnvSummary.cult_id == cult_id,
            EnvSummary.measure_date == measure_date,
        )
        .first()
    )

    if not summary:
        summary = EnvSummary(
            cult_id=cult_id,
            measure_date=measure_date,
        )
        db.session.add(summary)

    summary.daily_out_temp = float(agg[0]) if agg[0] is not None else None
    summary.daily_acc_solar = float(agg[1]) if agg[1] is not None else None
    summary.daily_rain_detection = int(agg[2]) if agg[2] is not None else None
    summary.daily_in_temp = float(agg[3]) if agg[3] is not None else None
    summary.daily_in_humidity = float(agg[4]) if agg[4] is not None else None
    summary.daily_in_co2 = float(agg[5]) if agg[5] is not None else None
    summary.daily_soil_temp = float(agg[6]) if agg[6] is not None else None
    summary.acc_temp = float(agg[7]) if agg[7] is not None else None
    summary.acc_solar = float(agg[8]) if agg[8] is not None else None

    return summary


def ingest_environment_one(data):
    env = build_environment_entity(data)

    db.session.add(env)
    db.session.flush()

    measure_date = env.measure_time.date()

    cleaned_count = rebuild_cleaned_for_day(env.cult_id, measure_date)
    upsert_env_summary_from_cleaned(env.cult_id, measure_date)

    db.session.commit()

    return {
        "env_id": env.env_id,
        "cult_id": env.cult_id,
        "measure_time": env.measure_time.strftime("%Y-%m-%d %H:%M:%S"),
        "measure_date": measure_date.strftime("%Y-%m-%d"),
        "cleaned_count": cleaned_count,
    }


def ingest_environment_bulk(data_list):
    if not data_list:
        raise ValueError("전송할 데이터가 없습니다.")

    env_list = []
    touched_dates = set()

    for data in data_list:
        env = build_environment_entity(data)
        env_list.append(env)
        touched_dates.add((env.cult_id, env.measure_time.date()))

    db.session.add_all(env_list)
    db.session.flush()

    cleaned_total = 0

    for cult_id, measure_date in touched_dates:
        cleaned_total += rebuild_cleaned_for_day(cult_id, measure_date)
        upsert_env_summary_from_cleaned(cult_id, measure_date)

    db.session.commit()

    return {
        "inserted_count": len(env_list),
        "updated_summary_count": len(touched_dates),
        "recleaned_count": cleaned_total,
    }