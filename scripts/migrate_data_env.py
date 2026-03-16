import pandas as pd
import os
import unicodedata
import re
import numpy as np
from smartfarm import db, create_app
from smartfarm.models import Farms, Cultivations, Environment


def migrate_environment():
    app = create_app()
    with app.app_context():
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.normpath(os.path.join(current_script_path, '..', 'data'))

        all_files = os.listdir(data_path)
        file_list = [f for f in all_files if '환경' in unicodedata.normalize('NFC', f) and f.endswith('.csv')]
        file_list.sort()

        print(f"📂 총 {len(file_list)}개의 파일을 처리합니다.")

        for file_name in file_list:
            display_name = unicodedata.normalize('NFC', file_name)
            year_match = re.search(r'\d{4}', display_name)
            if not year_match: continue
            file_year = int(year_match.group())

            print(f"🚀 [환경정보] {display_name} 처리 시작...")

            try:
                # --- 인코딩 자동 대응 ---
                try:
                    df = pd.read_csv(os.path.join(data_path, file_name), encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(os.path.join(data_path, file_name), encoding='cp949')

                # 컬럼명 정리 및 공백 제거
                df.columns = [c.strip() for c in df.columns]

                env_batch = []
                cult_cache = {}  # 동일 농가/작기 쿼리 중복 방지 캐시
                success_count = 0

                for _, row in df.iterrows():
                    try:
                        # 1. 행별 농가 번호와 작기 정보 추출 (생산 데이터 로직 이식)
                        f_num = int(float(row['farm_num']))
                        c_cycle = int(float(row['crop_cycle']))

                        # 2. 캐시 확인 또는 DB 조회
                        cache_key = f"{file_year}_{f_num}_{c_cycle}"
                        if cache_key in cult_cache:
                            cult_id = cult_cache[cache_key]
                        else:
                            target_farm_name = f"{file_year}_{f_num}"
                            farm = Farms.query.filter_by(farm_name=target_farm_name).first()
                            if not farm: continue

                            cult = Cultivations.query.filter_by(
                                farm_id=farm.farm_id,
                                crop_cycle=c_cycle,
                                survey_year=file_year
                            ).first()

                            if not cult: continue
                            cult_id = cult.cult_id
                            cult_cache[cache_key] = cult_id

                        # 3. 환경 데이터 객체 생성
                        new_env = Environment(
                            cult_id=cult_id,
                            measure_time=pd.to_datetime(row['measure_time']),
                            out_temp=float(row['out_temp']) if pd.notna(row['out_temp']) else None,
                            out_wind_direction=float(row['out_wind_direction']) if pd.notna(
                                row['out_wind_direction']) else None,
                            out_wind_speed=float(row['out_wind_speed']) if pd.notna(row['out_wind_speed']) else None,
                            out_solar_rad=float(row['out_solar_rad']) if pd.notna(row['out_solar_rad']) else None,
                            out_acc_solar_rad=float(row['out_acc_solar_rad']) if pd.notna(
                                row['out_acc_solar_rad']) else None,
                            rain_detection=int(float(row['rain_detection'])) if pd.notna(row['rain_detection']) else 0,
                            in_temp=float(row['in_temp']) if pd.notna(row['in_temp']) else None,
                            in_humidity=float(row['in_humidity']) if pd.notna(row['in_humidity']) else None,
                            in_co2=float(row['in_co2']) if pd.notna(row['in_co2']) else None,
                            soil_temp=float(row['soil_temp']) if pd.notna(row['soil_temp']) else None
                        )
                        env_batch.append(new_env)

                        # 4. 10,000건마다 벌크 삽입 (메모리 및 속도 효율)
                        if len(env_batch) >= 10000:
                            db.session.bulk_save_objects(env_batch)
                            db.session.commit()
                            success_count += len(env_batch)
                            env_batch = []

                    except Exception:
                        continue

                # 남은 데이터 처리
                if env_batch:
                    db.session.bulk_save_objects(env_batch)
                    db.session.commit()
                    success_count += len(env_batch)

                print(f"✅ {display_name} 적재 완료! ({success_count}건)")

            except Exception as e:
                db.session.rollback()
                print(f"❌ {display_name} 처리 중 에러: {e}")

        print("\n✨ 모든 환경 데이터 적재가 완료되었습니다!")


if __name__ == "__main__":
    migrate_environment()