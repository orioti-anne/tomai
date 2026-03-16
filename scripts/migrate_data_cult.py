import pandas as pd
import os
import unicodedata
from smartfarm import db, create_app
from smartfarm.models import Farms, Cultivations

def migrate_cultivations():
    app = create_app()
    with app.app_context():
        # 1. 경로 설정
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_script_path, '..', 'data')
        abs_data_path = os.path.abspath(data_path).strip()

        print(f"🔍 데이터 탐색 경로: [{abs_data_path}]")

        # 2. 파일 목록 필터링 (한글 NFD 대응)
        all_files = os.listdir(abs_data_path)
        file_list = []
        for f in all_files:
            nfc_f = unicodedata.normalize('NFC', f)
            if '재배정보' in nfc_f and nfc_f.endswith('.csv'):
                file_list.append(f)

        file_list.sort()

        if not file_list:
            print("❌ 처리할 CSV 파일을 찾지 못했습니다.")
            return

        print(f"📂 총 {len(file_list)}개의 파일에서 상세 정보를 추출합니다.")

        for file_name in file_list:
            file_path = os.path.join(abs_data_path, file_name)
            display_name = unicodedata.normalize('NFC', file_name)
            print(f"🚀 처리 중: {display_name}")

            try:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='cp949')

                success_count = 0
                for _, row in df.iterrows():
                    try:
                        # 3. 매칭 키 생성
                        s_year = int(float(row['survey_year']))
                        f_num = int(float(row['farm_num']))
                        target_farm_name = f"{s_year}_{f_num}"

                        # 4. 상위 테이블(Farms)에서 실제 farm_id 가져오기
                        farm = Farms.query.filter_by(farm_name=target_farm_name).first()

                        if not farm:
                            continue

                        # 날짜 및 숫자 처리
                        p_date = pd.to_datetime(row['planting_date']) if pd.notna(row['planting_date']) else None

                        # 5. Cultivations 객체 생성 (status='closed' 추가)
                        new_cult = Cultivations(
                            farm_id=farm.farm_id,
                            item=row['item'],
                            item_variety=row['item_variety'],
                            crop_cycle=int(float(row['crop_cycle'])),
                            planting_date=p_date,
                            planting_area=float(row['planting_area']),
                            planting_density=float(row['planting_density']),
                            house_type=row['house_type'],
                            house_form=row['house_form'],
                            survey_year=s_year,
                            status='closed'
                        )
                        db.session.add(new_cult)
                        success_count += 1
                    except Exception:
                        continue

                db.session.commit()
                print(f"✅ {display_name} 완료! ({success_count}건 매칭됨)")

            except Exception as e:
                db.session.rollback()
                print(f"❌ {display_name} 처리 중 에러: {e}")

        print("\n✨ 모든 상세 재배정보 이관 작업이 완료되었습니다!")

if __name__ == "__main__":
    migrate_cultivations()