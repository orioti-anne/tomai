import math
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import text

from smartfarm import db
from smartfarm.services.env_cleaning_service import rebuild_env_cleaned
from smartfarm.services.env_summary_service import rebuild_env_summary


RAW_COLUMNS = [
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
]

NUMERIC_COLUMNS = [
    "cult_id",
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
]


def _to_python_scalar(value):
    """
    Oracle 바인딩용 안전한 Python scalar 변환
    """
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


def normalize_environment_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSV / 센서 입력 공통 정규화
    """
    if df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    missing = [c for c in ["cult_id", "measure_time"] if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["measure_time"] = pd.to_datetime(df["measure_time"], errors="coerce")
    df = df[df["measure_time"].notna()].copy()

    if df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    for col in NUMERIC_COLUMNS:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("도", "", regex=False)
            .replace(
                {
                    "nan": None,
                    "None": None,
                    "": None,
                    "-": None,
                    ".": None,
                }
            )
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["cult_id"].notna()].copy()

    if df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    df["cult_id"] = df["cult_id"].astype(int)

    result = (
        df[RAW_COLUMNS]
        .drop_duplicates(subset=RAW_COLUMNS)
        .sort_values(["cult_id", "measure_time"])
        .reset_index(drop=True)
    )

    return result


def get_impacted_ranges(df: pd.DataFrame) -> list[dict]:
    """
    cult_id별 영향 날짜 범위 계산
    """
    if df.empty:
        return []

    tmp = df.copy()
    tmp["measure_date"] = pd.to_datetime(tmp["measure_time"]).dt.date

    grouped = (
        tmp.groupby("cult_id")["measure_date"]
        .agg(["min", "max"])
        .reset_index()
    )

    result = []
    for _, row in grouped.iterrows():
        result.append(
            {
                "cult_id": int(row["cult_id"]),
                "start_date": row["min"].strftime("%Y-%m-%d"),
                "end_date": row["max"].strftime("%Y-%m-%d"),
            }
        )

    return result


def replace_environment_raw(df: pd.DataFrame) -> int:
    """
    동일 cult_id + 날짜 범위의 raw 데이터를 삭제 후 다시 적재
    - delete: 일 단위 분할
    - insert: bulk execute
    - delete / insert 트랜잭션 분리
    """
    if df.empty:
        return 0

    impacted = get_impacted_ranges(df)

    rows = [
        {col: _to_python_scalar(val) for col, val in row.items()}
        for row in df.to_dict(orient="records")
    ]

    delete_sql = text(
        """
        DELETE FROM environment
        WHERE CULT_ID = :cult_id
          AND MEASURE_TIME >= :start_dt
          AND MEASURE_TIME < :end_dt
        """
    )

    insert_sql = text(
        """
        INSERT INTO environment (
            cult_id, measure_time,
            out_temp, out_wind_direction, out_wind_speed,
            out_solar_rad, out_acc_solar_rad, rain_detection,
            in_temp, in_humidity, in_co2, soil_temp,
            created_at
        ) VALUES (
            :cult_id, :measure_time,
            :out_temp, :out_wind_direction, :out_wind_speed,
            :out_solar_rad, :out_acc_solar_rad, :rain_detection,
            :in_temp, :in_humidity, :in_co2, :soil_temp,
            NOW()
        )
        """
    )

    print("replace_environment_raw delete 시작", flush=True)

    with db.engine.begin() as conn:
        for item in impacted:
            cult_id = item["cult_id"]
            current_dt = datetime.strptime(item["start_date"], "%Y-%m-%d")
            range_end = datetime.strptime(item["end_date"], "%Y-%m-%d") + timedelta(days=1)

            while current_dt < range_end:
                next_dt = current_dt + timedelta(days=1)

                conn.execute(
                    delete_sql,
                    {
                        "cult_id": cult_id,
                        "start_dt": current_dt,
                        "end_dt": next_dt,
                    },
                )

                current_dt = next_dt

    print("replace_environment_raw delete 완료", flush=True)
    print(f"replace_environment_raw insert 시작: {len(rows)}건", flush=True)

    with db.engine.begin() as conn:
        conn.execute(insert_sql, rows)

    print("replace_environment_raw insert 완료", flush=True)
    return len(rows)


def ingest_environment_data(df: pd.DataFrame) -> dict:
    """
    전체 파이프라인
    1) ENVIRONMENT 덮어쓰기 적재
    2) ENV_CLEANED 재생성
    3) ENV_SUMMARY 재생성
    """
    norm_df = normalize_environment_input(df)

    if norm_df.empty:
        return {
            "raw_inserted": 0,
            "cleaned_inserted": 0,
            "summary_rows": 0,
            "impacted": [],
        }

    impacted = get_impacted_ranges(norm_df)

    raw_inserted = replace_environment_raw(norm_df)

    cleaned_total = 0
    summary_total = 0

    for item in impacted:
        cult_id = item["cult_id"]
        start_date = item["start_date"]
        end_date = item["end_date"]

        cleaned_cnt = rebuild_env_cleaned(cult_id, start_date, end_date)
        summary_cnt = rebuild_env_summary(cult_id, start_date, end_date)

        cleaned_total += cleaned_cnt
        summary_total += summary_cnt

    return {
        "raw_inserted": raw_inserted,
        "cleaned_inserted": cleaned_total,
        "summary_rows": summary_total,
        "impacted": impacted,
    }