import pandas as pd
import os
import unicodedata
import re
from smartfarm import db, create_app
from smartfarm.models import Farms, Cultivations, Products


def migrate_products():
    app = create_app()
    with app.app_context():
        # 1. 경로 설정
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.normpath(os.path.join(current_script_path, '..', 'data'))

        if not os.path.exists(data_path):
            print(f"❌ 데이터 폴더를 찾을 수 없습니다: {data_path}")
            return

        all_files = os.listdir(data_path)
        # '생산'이 포함된 CSV 파일 필터링 (NFC 정규화 적용)
        file_list = [f for f in all_files if '생산' in unicodedata.normalize('NFC', f) and f.endswith('.csv')]
        file_list.sort()

        if not file_list:
            print("❌ 처리할 생산 정보 CSV 파일이 없습니다.")
            return

        print(f"📂 총 {len(file_list)}개의 파일을 처리합니다.")

        for file_name in file_list:
            file_path = os.path.join(data_path, file_name)
            display_name = unicodedata.normalize('NFC', file_name)

            # 2. 파일명에서 연도 추출 (예: 2022)
            year_match = re.search(r'\d{4}', display_name)
            if not year_match:
                print(f"⚠️ 파일명에서 연도를 찾을 수 없어 건너뜁니다: {display_name}")
                continue
            file_year = int(year_match.group())

            print(f"🚀 [생산정보] {display_name} 처리 시작...")

            try:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='cp949')

                success_count = 0
                for _, row in df.iterrows():
                    try:
                        # 3. 결측치 처리 (NaN이면 None으로 할당하여 DB에 NULL 입력)
                        t_qty = float(row['total_quantity']) if pd.notna(row['total_quantity']) else None
                        t_sales = float(row['total_sales']) if pd.notna(row['total_sales']) else None

                        # 날짜 변환
                        p_date = pd.to_datetime(row['production_date']) if pd.notna(row['production_date']) else None

                        # 4. FK 매칭 (Farms -> Cultivations)
                        f_num = int(float(row['farm_num']))
                        c_cycle = int(float(row['crop_cycle']))

                        target_farm_name = f"{file_year}_{f_num}"
                        farm = Farms.query.filter_by(farm_name=target_farm_name).first()

                        if not farm:
                            continue  # 농가 정보 없으면 스킵

                        cult = Cultivations.query.filter_by(
                            farm_id=farm.farm_id,
                            crop_cycle=c_cycle
                        ).first()

                        if not cult:
                            continue  # 재배 정보(작기) 매칭 안 되면 스킵

                        # 5. Products 객체 생성
                        new_prod = Products(
                            cult_id=cult.cult_id,
                            production_date=p_date,
                            total_quantity=t_qty,
                            total_sales=t_sales
                        )
                        db.session.add(new_prod)
                        success_count += 1

                    except Exception:
                        # 행 단위 에러는 무시하고 계속 진행
                        continue

                # 파일 단위 커밋
                db.session.commit()
                print(f"✅ {display_name} 완료! ({success_count}건 매칭 성공)")

            except Exception as e:
                db.session.rollback()
                print(f"❌ {display_name} 처리 중 치명적 에러: {e}")

        print("\n✨ 모든 생산 정보 이관 작업이 끝났습니다!")


if __name__ == "__main__":
    migrate_products()