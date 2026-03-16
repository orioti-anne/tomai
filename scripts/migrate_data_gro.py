import pandas as pd
import os
import unicodedata
import re
import numpy as np
from smartfarm import db, create_app
from smartfarm.models import Farms, Cultivations, Growth


def migrate_growth():
    app = create_app()
    with app.app_context():
        # 1. 경로 설정
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.normpath(os.path.join(current_script_path, '..', 'data'))

        if not os.path.exists(data_path):
            print(f"❌ 데이터 폴더를 찾을 수 없습니다: {data_path}")
            return

        all_files = os.listdir(data_path)
        # '생육'이 포함된 CSV 파일 필터링
        file_list = [f for f in all_files if '생육' in unicodedata.normalize('NFC', f) and f.endswith('.csv')]
        file_list.sort()

        if not file_list:
            print("❌ 처리할 생육 정보 CSV 파일이 없습니다.")
            return

        print(f"📂 총 {len(file_list)}개의 파일을 처리합니다.")

        for file_name in file_list:
            file_path = os.path.join(data_path, file_name)
            display_name = unicodedata.normalize('NFC', file_name)

            # 파일명에서 연도 추출
            year_match = re.search(r'\d{4}', display_name)
            if not year_match:
                print(f"⚠️ 연도 미검출 건너뜀: {display_name}")
                continue
            file_year = int(year_match.group())

            print(f"🚀 [생육정보] {display_name} 처리 시작...")

            try:
                # 인코딩 대응
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='cp949')

                # 전처리: 공백 제거 및 NaN 통일
                df = df.replace(r'^\s*$', np.nan, regex=True)
                df.columns = [c.strip() for c in df.columns]

                success_count = 0
                for _, row in df.iterrows():
                    try:
                        # 2. 필수 FK 매칭 정보 추출
                        f_num = int(float(row['farm_num']))
                        c_cycle = int(float(row['crop_cycle']))

                        target_farm_name = f"{file_year}_{f_num}"
                        farm = Farms.query.filter_by(farm_name=target_farm_name).first()

                        if not farm: continue

                        cult = Cultivations.query.filter_by(
                            farm_id=farm.farm_id,
                            crop_cycle=c_cycle,
                            survey_year=file_year
                        ).first()

                        if not cult: continue

                        # 3. 객체 생성 (컬럼 존재 여부를 체크하며 매핑)
                        # row.get('컬럼명')을 사용하면 해당 컬럼이 없을 때 None을 반환합니다.
                        new_growth = Growth(
                            cult_id=cult.cult_id,
                            inspect_date=pd.to_datetime(row['inspect_date']),
                            plant_num=int(float(row['plant_num'])) if pd.notna(row.get('plant_num')) else None,
                            branch_num=int(float(row['branch_num'])) if pd.notna(row.get('branch_num')) else None,
                            plant_height=float(row['plant_height']) if pd.notna(row.get('plant_height')) else None,
                            growth_length=float(row['growth_length']) if pd.notna(row.get('growth_length')) else None,
                            leaf_count=int(float(row['leaf_count'])) if pd.notna(row.get('leaf_count')) else None,
                            leaf_length=float(row['leaf_length']) if pd.notna(row.get('leaf_length')) else None,
                            leaf_width=float(row['leaf_width']) if pd.notna(row.get('leaf_width')) else None,
                            branch_width=float(row['branch_width']) if pd.notna(row.get('branch_width')) else None,
                            cluster_height=float(row['cluster_height']) if pd.notna(
                                row.get('cluster_height')) else None,
                            cluster_num=int(float(row['cluster_num'])) if pd.notna(row.get('cluster_num')) else None,
                            flowers_per_cluster=float(row['flowers_per_cluster']) if pd.notna(
                                row.get('flowers_per_cluster')) else None,
                            blooming_per_cluster=float(row['blooming_per_cluster']) if pd.notna(
                                row.get('blooming_per_cluster')) else None,
                            fruits_per_cluster=float(row['fruits_per_cluster']) if pd.notna(
                                row.get('fruits_per_cluster')) else None,

                            # 연도별로 있고 없을 수 있는 컬럼들 처리
                            blooming_group=float(row['blooming_group']) if 'blooming_group' in row and pd.notna(
                                row['blooming_group']) else None,
                            fruiting_group=float(row['fruiting_group']) if 'fruiting_group' in row and pd.notna(
                                row['fruiting_group']) else None,

                            # 2022년의 '비고' 컬럼을 DB의 'remarks'로 매핑
                            remarks=str(row['비고']) if '비고' in row and pd.notna(row['비고']) else None
                        )
                        db.session.add(new_growth)
                        success_count += 1

                    except Exception:
                        continue  # 행 단위 에러는 무시

                db.session.commit()
                print(f"✅ {display_name} 완료! ({success_count}건 적재 성공)")

            except Exception as e:
                db.session.rollback()
                print(f"❌ {display_name} 처리 중 치명적 에러: {e}")

        print("\n✨ 모든 생육 정보 이관 작업이 끝났습니다!")


if __name__ == "__main__":
    migrate_growth()