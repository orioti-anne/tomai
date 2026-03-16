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


def process_integrated_data(raw_df):
    """
    농가별/날짜별 데이터 정제 및 컬럼 필터링
    - 대상 컬럼: env_id, cult_id, measure_time, out_temp, out_wind_direction, out_wind_speed,
                out_solar_rad, out_acc_solar_rad, rain_detection, in_temp, in_humidity,
                in_co2, soil_temp, created_at, region_l1, region_l2,
                out_acc_solar_rad_status, date, hour
    """
    if raw_df.empty:
        return raw_df

    raw_df['measure_time'] = pd.to_datetime(raw_df['measure_time'])
    final_list = []
    all_cult_ids = sorted(raw_df['cult_id'].unique().tolist())

    for cid in all_cult_ids:
        df = raw_df[raw_df['cult_id'] == cid].copy()

        # 1. 중복값 처리
        df = df.sort_values('measure_time')
        df = df.drop_duplicates(subset=['measure_time'], keep='first').reset_index(drop=True)
        df['date'] = df['measure_time'].dt.date

        # 2. 상태값 초기화 (누적 일사량만 유지)
        df['out_acc_solar_rad_status'] = 0

        # 3. 누적 일사량(out_acc_solar_rad) 정밀 검증
        for date_val, group in df.groupby('date'):
            idx = group.index
            acc_vals = group['out_acc_solar_rad'].values
            to_remove = np.zeros(len(acc_vals), dtype=bool)

            if not np.all(pd.isna(acc_vals)):
                day_min = np.nanmin(acc_vals)
                day_max = np.nanmax(acc_vals)
                day_diff = day_max - day_min

                # [CASE A] 정체 및 리셋 부재 통합 감지
                # 1. 하루 상승폭이 10 미만으로 매우 적고 (정체)
                # 2. 동시에 하루 최솟값이 80을 초과한다면 (리셋 안됨)
                # 이 두 조건이 모두 해당되면 '신뢰할 수 없는 누적 데이터'로 판단하여 제거
                if day_diff < 10 and day_min > 80:
                    to_remove[:] = True
                else:
                    for i in range(1, len(acc_vals)):
                        prev_v = acc_vals[i - 1]
                        curr_v = acc_vals[i]
                        if pd.isna(curr_v) or pd.isna(prev_v): continue

                        # 역전 현상
                        if curr_v < prev_v:
                            if i + 1 < len(acc_vals):
                                next_v = acc_vals[i + 1]
                                if not pd.isna(next_v) and next_v > prev_v * 0.9:
                                    to_remove[i] = True

                        # 순간값 혼입 (변동 계수)
                        if i >= 2:
                            window = acc_vals[max(0, i - 2):i + 1]
                            window = window[~pd.isna(window)]
                            if len(window) >= 3:
                                cv = np.std(window) / (np.mean(window) + 1e-6)
                                if cv > 0.8:
                                    to_remove[i] = True

            # [CASE B] 물리적 한계치 초과 (5,000 이상)
            to_remove = to_remove | (group['out_acc_solar_rad'] > 5000)

            if to_remove.any():
                df.loc[idx[to_remove], 'out_acc_solar_rad'] = np.nan
                df.loc[idx[to_remove], 'out_acc_solar_rad_status'] = 9

        # 4. 마무리 정리 및 컬럼 필터링
        df['created_at'] = datetime.now()
        df['hour'] = df['measure_time'].dt.hour

        # 요청하신 19개 컬럼만 선택
        keep_cols = [
            'env_id', 'cult_id', 'measure_time', 'out_temp', 'out_wind_direction',
            'out_wind_speed', 'out_solar_rad', 'out_acc_solar_rad', 'rain_detection',
            'in_temp', 'in_humidity', 'in_co2', 'soil_temp', 'created_at',
            'region_l1', 'region_l2', 'out_acc_solar_rad_status', 'date', 'hour'
        ]

        # 존재하지 않는 컬럼이 있을 경우를 대비해 intersection 처리
        df = df[[c for c in keep_cols if c in df.columns]]

        final_list.append(df)

    return pd.concat(final_list, ignore_index=True)


if __name__ == "__main__":
    data_dir = os.path.join(project_root, 'data')
    if not os.path.exists(data_dir): os.makedirs(data_dir)

    app = create_app()

    with app.app_context():
        print("🔍 전체 농가(cult_id) 리스트 추출 중...")
        cid_query = text("SELECT DISTINCT cult_id FROM cultivations ORDER BY cult_id")
        all_cids = [row[0] for row in db.session.execute(cid_query)]
        print(f"✅ 총 {len(all_cids)}개의 농가가 확인되었습니다.")

        chunk_size = 50
        cid_chunks = [all_cids[i:i + chunk_size] for i in range(0, len(all_cids), chunk_size)]

        for chunk_idx, chunk in enumerate(cid_chunks, 1):
            print(f"\n🚀 [Part {chunk_idx}] 농가 {len(chunk)}개 처리 시작 (ID: {chunk[0]} ~ {chunk[-1]})")

            chunk_data_list = []
            cid_str = ", ".join(map(str, chunk))

            for year in range(2018, 2023):
                query = text(f"""
                    SELECT e.*, f.region_l1, f.region_l2
                    FROM environment e
                    JOIN cultivations c ON e.cult_id = c.cult_id
                    JOIN farms f ON c.farm_id = f.farm_id
                    WHERE e.cult_id IN ({cid_str})
                      AND e.measure_time >= TO_DATE('{year}-01-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
                      AND e.measure_time <  TO_DATE('{year + 1}-01-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
                    ORDER BY e.cult_id, e.measure_time
                """)

                year_df = pd.read_sql(query, db.engine)

                if not year_df.empty:
                    print(f"  📅 {year}년 데이터 로드 완료: {len(year_df):,}행", end='\r')
                    chunk_data_list.append(year_df)
                else:
                    print(f"  📅 {year}년 데이터 없음          ", end='\r')

            if chunk_data_list:
                full_chunk_df = pd.concat(chunk_data_list)
                print(f"\n  ✨ 데이터 정제 중 ({len(full_chunk_df):,} 행)...")

                cleaned_chunk_df = process_integrated_data(full_chunk_df)

                output_file = os.path.join(data_dir, f'env_cleaned_part_{chunk_idx}.csv')
                cleaned_chunk_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"  💾 저장 완료: {output_file}")
            else:
                print(f"\n  ⚠️ 해당 농가 그룹의 데이터가 없습니다.")

    print("\n🏁 모든 파트의 전처리가 완료되었습니다!")