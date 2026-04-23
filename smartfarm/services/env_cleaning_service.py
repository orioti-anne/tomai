import math
import pandas as pd
import numpy as np
from sqlalchemy import text
from smartfarm import db


def _to_python_scalar(value):
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, float) and math.isnan(value):
        return None

    return value


def process_chunk_final_v4(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    df["measure_time"] = pd.to_datetime(df["measure_time"], errors="coerce")
    df = df[df["measure_time"].notna()].copy()

    df["measure_date"] = df["measure_time"].dt.normalize()
    df["measure_hour"] = df["measure_time"].dt.hour

    df["out_acc_solar_rad_status"] = 0
    df["in_temp_status"] = 0
    df["in_humidity_status"] = 0
    df["in_co2_status"] = 0

    df = (
        df.sort_values(["cult_id", "measure_time"])
        .drop_duplicates(subset=["cult_id", "measure_time"], keep="first")
        .copy()
    )

    final_list = []

    for (_, _), day_df in df.groupby(["cult_id", "measure_date"]):
        day_df = day_df.sort_values("measure_time").copy()

        env_ranges = {
            "in_temp": (-10, 55, "in_temp_status"),
            "in_humidity": (0, 100, "in_humidity_status"),
            "in_co2": (200, 3500, "in_co2_status"),
        }

        for col, (min_val, max_val, status_col) in env_ranges.items():
            if col not in day_df.columns:
                day_df[col] = np.nan

            outlier_mask = (day_df[col] < min_val) | (day_df[col] > max_val)
            if outlier_mask.any():
                day_df.loc[outlier_mask, status_col] = 9
                day_df.loc[outlier_mask, col] = np.nan

            missing_before = day_df[col].isna()
            if missing_before.any():
                day_df[col] = day_df[col].interpolate(method="linear", limit_direction="both")
                filled_mask = missing_before & day_df[col].notna()
                day_df.loc[filled_mask & (day_df[status_col] == 0), status_col] = 5

        for col in ["out_solar_rad", "out_acc_solar_rad"]:
            if col not in day_df.columns:
                day_df[col] = 0

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

        is_mismatch = abs(real_diff - calc_max) > (calc_max * adaptive_tolerance) if calc_max > 0 else False
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
        "env_id",
        "cult_id",
        "measure_time",
        "out_temp",
        "out_wind_direction",
        "out_wind_speed",
        "out_solar_rad",
        "out_acc_solar_rad",
        "rain_detection",
        "in_temp",
        "in_humidity",
        "in_co2",
        "soil_temp",
        "out_acc_solar_rad_status",
        "in_temp_status",
        "in_humidity_status",
        "in_co2_status",
        "measure_date",
        "measure_hour",
    ]

    for col in target_columns:
        if col not in processed_df.columns:
            processed_df[col] = np.nan

    return processed_df[target_columns]


def rebuild_env_cleaned(cult_id: int, start_date: str, end_date: str) -> int:
    print("cleaned 1. raw 조회 시작")

    raw_query = text("""
        SELECT *
        FROM environment
        WHERE CULT_ID = :cult_id
          AND measure_time >= :start_date::date
          AND measure_time < :end_date::date + INTERVAL '1 day'
        ORDER BY CULT_ID, MEASURE_TIME
    """)

    raw_df = pd.read_sql(
        raw_query,
        db.engine,
        params={
            "cult_id": cult_id,
            "start_date": start_date,
            "end_date": end_date,
        },
    )

    print("cleaned 2. raw 조회 완료:", len(raw_df))

    with db.engine.begin() as conn:
        print("cleaned 3. 기존 cleaned 삭제 시작")
        conn.execute(text("""
            DELETE FROM env_cleaned
            WHERE CULT_ID = :cult_id
              AND measure_date BETWEEN :start_date::date
                                   AND :end_date::date
        """), {
            "cult_id": cult_id,
            "start_date": start_date,
            "end_date": end_date,
        })
        print("cleaned 4. 기존 cleaned 삭제 완료")

    if raw_df.empty:
        return 0

    print("cleaned 5. 전처리 시작")
    cleaned_df = process_chunk_final_v4(raw_df)
    print("cleaned 6. 전처리 완료:", len(cleaned_df))

    if cleaned_df.empty:
        return 0

    cleaned_rows = [
        {col: _to_python_scalar(val) for col, val in row.items()}
        for row in cleaned_df.to_dict(orient="records")
    ]

    insert_sql = text("""
        INSERT INTO env_cleaned (
            env_id, cult_id, measure_time,
            out_temp, out_wind_direction, out_wind_speed,
            out_solar_rad, out_acc_solar_rad, rain_detection,
            in_temp, in_humidity, in_co2, soil_temp,
            out_acc_solar_rad_status, in_temp_status, in_humidity_status, in_co2_status,
            measure_date, measure_hour, created_at
        ) VALUES (
            :env_id, :cult_id, :measure_time,
            :out_temp, :out_wind_direction, :out_wind_speed,
            :out_solar_rad, :out_acc_solar_rad, :rain_detection,
            :in_temp, :in_humidity, :in_co2, :soil_temp,
            :out_acc_solar_rad_status, :in_temp_status, :in_humidity_status, :in_co2_status,
            :measure_date, :measure_hour, NOW()
        )
    """)

    with db.engine.begin() as conn:
        print("cleaned 7. insert 시작")
        for i, row in enumerate(cleaned_rows):
            try:
                conn.execute(insert_sql, row)
                if i % 100 == 0:
                    print(f"cleaned insert 진행중: {i}")
            except Exception as e:
                print("cleaned 실패 row index:", i)
                print("cleaned 실패 row:", row)
                print("cleaned 에러:", e)
                raise
        print("cleaned 8. insert 완료")

    return len(cleaned_rows)