from sqlalchemy import text
from smartfarm import db


def rebuild_env_summary(cult_id: int, start_date: str, end_date: str) -> int:
    """
    env_cleaned -> env_summary 재생성
    지정 cult_id / 날짜 범위만 삭제 후 재적재
    """

    with db.engine.begin() as conn:
        # 1. 기존 summary 삭제
        conn.execute(text("""
            DELETE FROM env_summary
            WHERE cult_id = :cult_id
              AND measure_date BETWEEN :start_date::date AND :end_date::date
        """), {"cult_id": cult_id, "start_date": start_date, "end_date": end_date})

        # 2. summary insert
        conn.execute(text("""
            INSERT INTO env_summary (
                cult_id, measure_date,
                daily_out_temp, daily_acc_solar, daily_rain_detection,
                daily_in_temp, daily_in_humidity, daily_in_co2, daily_soil_temp,
                acc_temp, acc_solar
            )
            SELECT
                cult_id, m_date,
                avg_out_temp, target_acc_solar, max_rain,
                avg_in_temp, avg_in_humidity, avg_in_co2, avg_soil_temp,
                running_acc_temp, running_acc_solar
            FROM (
                WITH daily_base AS (
                    SELECT
                        e.cult_id,
                        e.measure_date::date AS m_date,
                        c.planting_date::date AS p_date,
                        AVG(e.out_temp) AS avg_out_temp,
                        MAX(CASE
                            WHEN e.measure_hour BETWEEN 14 AND 23
                            THEN e.out_acc_solar_rad
                            ELSE 0
                        END) AS target_acc_solar,
                        MAX(e.rain_detection) AS max_rain,
                        AVG(e.in_temp) AS avg_in_temp,
                        AVG(e.in_humidity) AS avg_in_humidity,
                        AVG(e.in_co2) AS avg_in_co2,
                        AVG(e.soil_temp) AS avg_soil_temp
                    FROM env_cleaned e
                    LEFT JOIN cultivations c ON e.cult_id = c.cult_id
                    WHERE e.cult_id = :cult_id
                      AND e.measure_date BETWEEN :start_date::date AND :end_date::date
                    GROUP BY e.cult_id, e.measure_date::date, c.planting_date::date
                ),
                accumulated_calc AS (
                    SELECT
                        db.*,
                        CASE
                            WHEN db.m_date >= db.p_date THEN
                                SUM(CASE WHEN db.m_date >= db.p_date THEN db.avg_in_temp ELSE 0 END)
                                OVER (PARTITION BY db.cult_id ORDER BY db.m_date)
                            ELSE NULL
                        END AS running_acc_temp,
                        CASE
                            WHEN db.m_date >= db.p_date THEN
                                SUM(CASE WHEN db.m_date >= db.p_date THEN db.target_acc_solar ELSE 0 END)
                                OVER (PARTITION BY db.cult_id ORDER BY db.m_date)
                            ELSE NULL
                        END AS running_acc_solar
                    FROM daily_base db
                )
                SELECT * FROM accumulated_calc
            ) sub
        """), {"cult_id": cult_id, "start_date": start_date, "end_date": end_date})

        # 3. 0값 후처리
        conn.execute(text("""
            UPDATE env_summary
            SET daily_acc_solar = NULL
            WHERE cult_id = :cult_id
              AND measure_date BETWEEN :start_date::date AND :end_date::date
              AND daily_acc_solar = 0
        """), {"cult_id": cult_id, "start_date": start_date, "end_date": end_date})

        conn.execute(text("""
            UPDATE env_summary
            SET acc_solar = NULL
            WHERE cult_id = :cult_id
              AND measure_date BETWEEN :start_date::date AND :end_date::date
              AND acc_solar = 0
        """), {"cult_id": cult_id, "start_date": start_date, "end_date": end_date})

    with db.engine.begin() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) FROM env_summary
            WHERE cult_id = :cult_id
              AND measure_date BETWEEN :start_date::date AND :end_date::date
        """), {"cult_id": cult_id, "start_date": start_date, "end_date": end_date}).scalar()

    return int(result or 0)
