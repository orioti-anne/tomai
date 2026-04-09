from sqlalchemy import text
from smartfarm import db


def rebuild_env_summary(cult_id: int, start_date: str, end_date: str) -> int:
    """
    ENV_CLEANED -> ENV_SUMMARY 재생성
    지정 cult_id / 날짜 범위만 삭제 후 재적재
    """

    with db.engine.begin() as conn:
        # 1. 기존 summary 삭제
        conn.execute(text("""
            DELETE FROM ENV_SUMMARY
            WHERE CULT_ID = :cult_id
              AND MEASURE_DATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                                   AND TO_DATE(:end_date, 'YYYY-MM-DD')
        """), {
            "cult_id": cult_id,
            "start_date": start_date,
            "end_date": end_date,
        })

        # 2. summary insert
        conn.execute(text("""
            INSERT INTO ENV_SUMMARY (
                ENVSU_ID, CULT_ID, MEASURE_DATE,
                DAILY_OUT_TEMP, DAILY_ACC_SOLAR, DAILY_RAIN_DETECTION,
                DAILY_IN_TEMP, DAILY_IN_HUMIDITY, DAILY_IN_CO2, DAILY_SOIL_TEMP,
                ACC_TEMP, ACC_SOLAR
            )
            SELECT
                SEQ_ENVSU_ID.NEXTVAL,
                CULT_ID, M_DATE,
                AVG_OUT_TEMP, TARGET_ACC_SOLAR, MAX_RAIN,
                AVG_IN_TEMP, AVG_IN_HUMIDITY, AVG_IN_CO2, AVG_SOIL_TEMP,
                RUNNING_ACC_TEMP, RUNNING_ACC_SOLAR
            FROM (
                WITH DAILY_BASE AS (
                    SELECT
                        E.CULT_ID,
                        TRUNC(E.MEASURE_DATE) AS M_DATE,
                        TRUNC(C.PLANTING_DATE) AS P_DATE,
                        AVG(E.OUT_TEMP) AS AVG_OUT_TEMP,
                        MAX(CASE
                            WHEN E.MEASURE_HOUR BETWEEN 14 AND 23
                            THEN E.OUT_ACC_SOLAR_RAD
                            ELSE 0
                        END) AS TARGET_ACC_SOLAR,
                        MAX(E.RAIN_DETECTION) AS MAX_RAIN,
                        AVG(E.IN_TEMP) AS AVG_IN_TEMP,
                        AVG(E.IN_HUMIDITY) AS AVG_IN_HUMIDITY,
                        AVG(E.IN_CO2) AS AVG_IN_CO2,
                        AVG(E.SOIL_TEMP) AS AVG_SOIL_TEMP
                    FROM ENV_CLEANED E
                    LEFT JOIN CULTIVATIONS C
                        ON E.CULT_ID = C.CULT_ID
                    WHERE E.CULT_ID = :cult_id
                      AND E.MEASURE_DATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                                             AND TO_DATE(:end_date, 'YYYY-MM-DD')
                    GROUP BY
                        E.CULT_ID,
                        TRUNC(E.MEASURE_DATE),
                        TRUNC(C.PLANTING_DATE)
                ),
                ACCUMULATED_CALC AS (
                    SELECT
                        DB.*,
                        CASE
                            WHEN DB.M_DATE >= DB.P_DATE THEN
                                SUM(CASE
                                    WHEN DB.M_DATE >= DB.P_DATE THEN DB.AVG_IN_TEMP
                                    ELSE 0
                                END) OVER (
                                    PARTITION BY DB.CULT_ID
                                    ORDER BY DB.M_DATE
                                )
                            ELSE NULL
                        END AS RUNNING_ACC_TEMP,
                        CASE
                            WHEN DB.M_DATE >= DB.P_DATE THEN
                                SUM(CASE
                                    WHEN DB.M_DATE >= DB.P_DATE THEN DB.TARGET_ACC_SOLAR
                                    ELSE 0
                                END) OVER (
                                    PARTITION BY DB.CULT_ID
                                    ORDER BY DB.M_DATE
                                )
                            ELSE NULL
                        END AS RUNNING_ACC_SOLAR
                    FROM DAILY_BASE DB
                )
                SELECT * FROM ACCUMULATED_CALC
            )
        """), {
            "cult_id": cult_id,
            "start_date": start_date,
            "end_date": end_date,
        })

        # 3. 0값 후처리
        conn.execute(text("""
            UPDATE ENV_SUMMARY
            SET DAILY_ACC_SOLAR = NULL
            WHERE CULT_ID = :cult_id
              AND MEASURE_DATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                                   AND TO_DATE(:end_date, 'YYYY-MM-DD')
              AND DAILY_ACC_SOLAR = 0
        """), {
            "cult_id": cult_id,
            "start_date": start_date,
            "end_date": end_date,
        })

        conn.execute(text("""
            UPDATE ENV_SUMMARY
            SET ACC_SOLAR = NULL
            WHERE CULT_ID = :cult_id
              AND MEASURE_DATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                                   AND TO_DATE(:end_date, 'YYYY-MM-DD')
              AND ACC_SOLAR = 0
        """), {
            "cult_id": cult_id,
            "start_date": start_date,
            "end_date": end_date,
        })

    count_sql = text("""
        SELECT COUNT(*)
        FROM ENV_SUMMARY
        WHERE CULT_ID = :cult_id
          AND MEASURE_DATE BETWEEN TO_DATE(:start_date, 'YYYY-MM-DD')
                               AND TO_DATE(:end_date, 'YYYY-MM-DD')
    """)

    with db.engine.begin() as conn:
        result = conn.execute(count_sql, {
            "cult_id": cult_id,
            "start_date": start_date,
            "end_date": end_date,
        }).scalar()

    return int(result or 0)