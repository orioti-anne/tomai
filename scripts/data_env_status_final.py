import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import text, types
from datetime import datetime

# [1] 경로 설정 (Flask 프로젝트 구조에 맞게 설정)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app


def process_chunk_final_v4(df):
    """
    스마트팜 환경 데이터 통합 정제 로직 v4.2
    - 내부 환경 및 누적 일사량 보정
    - DB 컬럼명 직접 사용 (measure_date, measure_hour)
    """
    if df.empty:
        return df

    # [개선] 처음부터 DB 컬럼명인 measure_date, measure_hour로 생성
    df['measure_time'] = pd.to_datetime(df['measure_time'])
    df['measure_date'] = df['measure_time'].dt.normalize()
    df['measure_hour'] = df['measure_time'].dt.hour

    # 상태 코드 초기화 (0: 정상)
    df['out_acc_solar_rad_status'] = 0
    df['in_temp_status'] = 0
    df['in_humidity_status'] = 0
    df['in_co2_status'] = 0

    # 중복 제거 (농가별 동일 시간 데이터 방지)
    df = df.sort_values(['cult_id', 'measure_time']).drop_duplicates(
        subset=['cult_id', 'measure_time'], keep='first'
    )

    final_list = []

    # 농가별, 날짜별 그룹핑 처리
    for (cid, d_val), day_df in df.groupby(['cult_id', 'measure_date']):
        day_df = day_df.copy()

        # --- [STEP 1] 내부 환경 데이터 정제 (온도, 습도, CO2) ---
        env_ranges = {
            'in_temp': (-10, 55, 'in_temp_status'),
            'in_humidity': (0, 100, 'in_humidity_status'),
            'in_co2': (200, 3500, 'in_co2_status')
        }

        for col, (min_val, max_val, status_col) in env_ranges.items():
            # 이상치 탐지 (Status 9)
            outlier_mask = (day_df[col] < min_val) | (day_df[col] > max_val)
            if outlier_mask.any():
                day_df.loc[outlier_mask, status_col] = 9
                day_df.loc[outlier_mask, col] = np.nan

            # 결측치 보간 (Status 5)
            missing_before = day_df[col].isna()
            if missing_before.any():
                day_df[col] = day_df[col].interpolate(method='linear', limit_direction='both')
                filled_mask = missing_before & day_df[col].notna()
                day_df.loc[filled_mask & (day_df[status_col] == 0), status_col] = 5

        # --- [STEP 2] 누적 일사량 정제 (물리 법칙 기반) ---
        # 야간 시간대(19시~06시) 일사량 강제 제로화
        night_mask = (day_df['measure_hour'] >= 19) | (day_df['measure_hour'] <= 6)
        day_df.loc[night_mask & (day_df['out_solar_rad'] > 20), 'out_solar_rad'] = 0

        # 이론적 누적치 계산 (물리적 적분)
        time_diff = day_df['measure_time'].diff().dt.total_seconds().fillna(0)
        day_df['calc_acc'] = (day_df['out_solar_rad'] * time_diff / 10000).cumsum()

        # 보정 로직 판단을 위한 변수들
        real_diff = day_df['out_acc_solar_rad'].max() - day_df['out_acc_solar_rad'].min()
        calc_max = day_df['calc_acc'].max()
        solar_mean = day_df['out_solar_rad'].mean()
        cv = (day_df['out_solar_rad'].std() / solar_mean) if solar_mean > 0 else 0
        base_tol = 0.15 if day_df['measure_hour'].between(10, 15).any() else 0.25
        adaptive_tolerance = base_tol + min(cv * 0.5, 0.15)

        # 전면 보정 조건 (Status 6)
        is_mismatch = (abs(real_diff - calc_max) > (calc_max * adaptive_tolerance))
        is_stagnant = (calc_max > 5 and real_diff < 5)
        is_over_limit = (day_df['out_acc_solar_rad'] > 4000).any()

        if is_mismatch or is_stagnant or is_over_limit:
            day_df['out_acc_solar_rad'] = day_df['calc_acc']
            day_df['out_acc_solar_rad_status'] = 6

        # 결측 보간 (Status 5)
        missing_acc_before = day_df['out_acc_solar_rad'].isna()
        if missing_acc_before.any():
            day_df.loc[missing_acc_before, 'out_acc_solar_rad'] = day_df['calc_acc']
            filled_acc_mask = missing_acc_before & day_df['out_acc_solar_rad'].notna()
            day_df.loc[filled_acc_mask & (day_df['out_acc_solar_rad_status'] == 0), 'out_acc_solar_rad_status'] = 5

        final_list.append(day_df)

    if not final_list: return pd.DataFrame()
    processed_df = pd.concat(final_list, ignore_index=True)

    target_columns = [
        'env_id', 'cult_id', 'measure_time', 'out_temp',
        'out_wind_direction', 'out_wind_speed', 'out_solar_rad',
        'out_acc_solar_rad', 'rain_detection', 'in_temp',
        'in_humidity', 'in_co2', 'soil_temp',
        'out_acc_solar_rad_status', 'in_temp_status', 'in_humidity_status', 'in_co2_status',
        'measure_date', 'measure_hour'
    ]

    return processed_df[target_columns]


if __name__ == "__main__":
    app = create_app()

    with app.app_context():
        # 1. 처리할 전체 농가 목록 조회
        all_cids_query = text("SELECT DISTINCT cult_id FROM environment ORDER BY cult_id")
        all_cids = [row[0] for row in db.session.execute(all_cids_query).fetchall()]

        batch_size = 10
        total_batches = (len(all_cids) - 1) // batch_size + 1

        for i in range(0, len(all_cids), batch_size):
            batch_cids = all_cids[i: i + batch_size]
            cid_str = ", ".join(map(str, batch_cids))
            current_batch = i // batch_size + 1

            print(f"\n 배치 [{current_batch}/{total_batches}] 시작: cult_id {batch_cids[0]} ~ {batch_cids[-1]}")

            # 2. 원본 데이터 로드
            query = text(f"SELECT * FROM environment WHERE cult_id IN ({cid_str}) ORDER BY cult_id, measure_time")
            batch_df = pd.read_sql(query, db.engine)

            if not batch_df.empty:
                # 3. 데이터 가공
                print(f"   ㄴ 가공 중...")
                refined_df = process_chunk_final_v4(batch_df)

                # 4. DB(ENV_CLEANED)로 즉시 적재
                if not refined_df.empty:
                    print(f"   ㄴ DB 적재 중 (Bulk Insert)...")
                    try:
                        refined_df.to_sql(
                            name='env_cleaned',
                            con=db.engine,
                            if_exists='append',
                            index=False,
                            chunksize=1000,
                            dtype={
                                'measure_time': types.TIMESTAMP,
                                'measure_date': types.DATE
                            }
                        )
                        print(f" 배치 적재 완료: {len(refined_df):,} rows")
                    except Exception as e:
                        print(f" DB 적재 실패 (배치 {current_batch}): {e}")

    print("\n 모든 데이터 정제 및 DB 이관 작업이 종료되었습니다.")