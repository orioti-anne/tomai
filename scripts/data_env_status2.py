import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import text
from datetime import datetime

# [1] 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app


def process_chunk_final_v3(df):
    """
    물리 법칙 기반 정제 로직 v3
    - 밤 시간대(19-06) 순간 일사량 20 초과 시 0 처리
    - 하루 누적치 증가량(diff) 및 절대치(any > 4000) 검사 강화
    """
    if df.empty: return df

    df['measure_time'] = pd.to_datetime(df['measure_time'])
    df['date'] = df['measure_time'].dt.date
    df['hour'] = df['measure_time'].dt.hour
    df['out_acc_solar_rad_status'] = 0
    df['created_at'] = datetime.now()

    # 중복 제거 및 정렬
    df = df.sort_values(['cult_id', 'measure_time']).drop_duplicates(subset=['cult_id', 'measure_time'], keep='first')

    final_list = []
    for (cid, d_val), day_df in df.groupby(['cult_id', 'date']):
        day_df = day_df.copy()

        # 1. 밤 시간대 순간 일사량 노이즈 제거
        night_mask = (day_df['hour'] >= 19) | (day_df['hour'] <= 6)
        day_df.loc[night_mask & (day_df['out_solar_rad'] > 20), 'out_solar_rad'] = 0

        # 2. 이론적 누적치 재계산
        time_diff = day_df['measure_time'].diff().dt.total_seconds().fillna(0)
        day_df['calc_acc'] = (day_df['out_solar_rad'] * time_diff / 10000).cumsum()

        # 3. 보정 조건 판단
        real_day_max = day_df['out_acc_solar_rad'].max()
        real_day_min = day_df['out_acc_solar_rad'].min()
        real_diff = real_day_max - real_day_min
        calc_max = day_df['calc_acc'].max()

        # [조건 A] 하루 증가량이 계산값과 20% 이상 차이남
        is_mismatch = (abs(real_diff - calc_max) > (calc_max * 0.2))
        # [조건 B] 해가 떴는데 수치가 거의 안 변함 (정체)
        is_stagnant = (calc_max > 5 and real_diff < 5)
        # [조건 C] 하루 중 단 한 번이라도 절대 수치가 4,000을 초과함 (비정상 고점)
        is_over_limit = (day_df['out_acc_solar_rad'] > 4000).any()

        if is_mismatch or is_stagnant or is_over_limit:
            day_df['out_acc_solar_rad'] = day_df['calc_acc']
            day_df['out_acc_solar_rad_status'] = 6

            # 4. 결측치 구간 채움
        missing_mask = day_df['out_acc_solar_rad'].isna() & day_df['out_solar_rad'].notna()
        if missing_mask.any():
            day_df.loc[missing_mask & (day_df['out_acc_solar_rad_status'] == 0), 'out_acc_solar_rad_status'] = 5
            day_df.loc[missing_mask, 'out_acc_solar_rad'] = day_df['calc_acc']

        final_list.append(day_df)

    processed_df = pd.concat(final_list, ignore_index=True)

    target_columns = [
        'env_id', 'cult_id', 'measure_time', 'out_temp',
        'out_wind_direction', 'out_wind_speed', 'out_solar_rad',
        'out_acc_solar_rad', 'rain_detection', 'in_temp',
        'in_humidity', 'in_co2', 'soil_temp',
        'out_acc_solar_rad_status', 'date', 'hour', 'created_at'
    ]

    for col in target_columns:
        if col not in processed_df.columns:
            processed_df[col] = np.nan

    return processed_df[target_columns]


if __name__ == "__main__":
    data_dir = os.path.join(project_root, 'data')
    if not os.path.exists(data_dir): os.makedirs(data_dir)

    app = create_app()

    with app.app_context():
        # 1. 전체 농가 ID 리스트 가져오기
        all_cids_query = text("SELECT DISTINCT cult_id FROM environment ORDER BY cult_id")
        all_cids = [row[0] for row in db.session.execute(all_cids_query).fetchall()]

        # 2. 45개씩 끊어서 처리
        batch_size = 45
        for i in range(0, len(all_cids), batch_size):
            batch_cids = all_cids[i: i + batch_size]
            start_id = batch_cids[0]
            end_id = batch_cids[-1]
            cid_str = ", ".join(map(str, batch_cids))

            print(f"\n📦 배치 처리 시작: cult_id {start_id} ~ {end_id}")

            query = text(f"""
                SELECT * FROM environment 
                WHERE cult_id IN ({cid_str})
                ORDER BY cult_id, measure_time
            """)

            # [최적화] 45개 농가 데이터를 한 번에 메모리에 로드 (날짜 잘림 방지)
            print(f"   ㄴ 데이터 로드 중...", end='\r')
            batch_df = pd.read_sql(query, db.engine)

            if not batch_df.empty:
                print(f"   ㄴ 데이터 정제 중...", end='\r')
                final_batch_df = process_chunk_final_v3(batch_df)

                # 파일 저장
                filename = f'env_cleaned_{start_id}_{end_id}.csv'
                output_path = os.path.join(data_dir, filename)
                final_batch_df.to_csv(output_path, index=False, encoding='utf-8-sig')

                print(f"\n✅ 저장 완료: {output_path}")
                print(f"📊 보정(6): {len(final_batch_df[final_batch_df['out_acc_solar_rad_status'] == 6]):,}건 | "
                      f"채움(5): {len(final_batch_df[final_batch_df['out_acc_solar_rad_status'] == 5]):,}건")

    print("\n🏁 모든 배치 작업이 종료되었습니다.")