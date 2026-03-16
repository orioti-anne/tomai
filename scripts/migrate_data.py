import pandas as pd
import os
import unicodedata
from smartfarm import db, create_app
from smartfarm.models import Farms


def migrate_farms_only():
    app = create_app()
    with app.app_context():
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_script_path, '..', 'data')
        abs_data_path = os.path.normpath(data_path).strip()

        all_files = os.listdir(abs_data_path)
        file_list = [f for f in all_files if '재배정보' in unicodedata.normalize('NFC', f) and f.endswith('.csv')]
        file_list.sort()

        for file_name in file_list:
            file_path = os.path.join(abs_data_path, file_name)
            print(f"🚀 [농가 등록] 처리 중: {unicodedata.normalize('NFC', file_name)}")

            try:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='cp949')

                new_farms_count = 0
                for index, row in df.iterrows():
                    s_year = int(float(row['survey_year']))
                    f_num = int(float(row['farm_num']))
                    f_name = f"{s_year}_{f_num}"

                    # [핵심] DB에 이미 해당 농장 이름이 있는지 체크
                    existing_farm = Farms.query.filter_by(farm_name=f_name).first()

                    if existing_farm:
                        # 이미 등록된 농장이면 상세 정보(Cultivations)에서 참조할 수 있으므로 건너뜀
                        continue

                    new_farm = Farms(
                        user_id=0,
                        farm_name=f_name,
                        farm_num=f_num,
                        region_l1=row['region_l1'],
                        region_l2=row['region_l2'],
                        total_area=float(row['total_area']),
                        first_survey_year=s_year
                    )
                    db.session.add(new_farm)
                    new_farms_count += 1

                db.session.commit()
                print(f"✅ {file_name} 완료! (새로 등록된 농가: {new_farms_count}건)")

            except Exception as e:
                db.session.rollback()
                print(f"❌ {file_name} 에러: {e}")

        print("\n✨ 유니크한 농가 데이터 이관이 완료되었습니다.")


if __name__ == "__main__":
    migrate_farms_only()