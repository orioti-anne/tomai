import os
import requests
import oracledb
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
ORACLE_PATH = "/Users/yunju/oracle/instantclient_23_3"
DB_CONFIG = {'user': 'SMART', 'password': 'tiger', 'dsn': 'localhost:1521/xe'}

API_KEY = "d9c4bd57-6f86-417e-85f0-4db5254fe40e"
USER_ID = "7367"
BASE_URL = "http://www.kamis.or.kr/service/price/xml.do"


def run_final_recovery():
    try:
        oracledb.init_oracle_client(lib_dir=ORACLE_PATH)
        conn = oracledb.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("🚀 [최종] 진짜 완숙토마토 데이터 적재를 시작합니다.")
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        return

    # 시작 날짜와 종료 날짜 설정
    start_date = datetime(2021, 1, 1)
    end_date = datetime.now() - timedelta(days=1)
    current_date = start_date

    while current_date <= end_date:
        d_str = current_date.strftime('%Y-%m-%d')

        params = {
            'action': 'dailyPriceByCategoryList',
            'p_product_cls_code': '02',
            'p_item_category_code': '200',
            'p_item_code': '225',
            'p_kind_code': '11',
            'p_country_code': '1101',
            'p_regday': d_str,
            'p_convert_kg_yn': 'N',  # 수동 계산이 정확함
            'p_cert_key': API_KEY,
            'p_cert_id': USER_ID,
            'p_returntype': 'json'
        }

        try:
            res = requests.get(BASE_URL, params=params, timeout=10)
            data = res.json()

            if data.get('data') and isinstance(data['data'].get('item'), list):
                all_items = data['data']['item']

                # 정밀 필터링: 배추, 방울, 대추 제외
                ripe_tomatoes = [
                    item for item in all_items
                    if ('토마토' in item.get('item_name', '') or '토마토' in item.get('kind_name', ''))
                       and '방울' not in item.get('kind_name', '')
                       and '대추' not in item.get('kind_name', '')
                       and '포기' not in item.get('unit', '')
                ]

                if ripe_tomatoes:
                    # '상품' 등급 우선 선택 (없으면 첫 번째 항목)
                    target = next((i for i in ripe_tomatoes if i.get('rank') == '상품'), ripe_tomatoes[0])

                    price_str = target.get('dpr1', '0').replace(',', '')
                    if price_str == '-':
                        current_date += timedelta(days=1)
                        continue

                    raw_price = float(price_str)
                    unit_str = target.get('unit', '5kg')
                    grade = target.get('rank', '상품')

                    # 1kg 환산
                    if '5kg' in unit_str:
                        price_per_kg = raw_price / 5
                    elif '10kg' in unit_str:
                        price_per_kg = raw_price / 10
                    elif '20kg' in unit_str:
                        price_per_kg = raw_price / 20
                    else:
                        price_per_kg = raw_price  # 이미 1kg인 경우

                    # DB INSERT
                    sql = """
                          INSERT INTO SMART.KAMIS_TOMATO_PRICE
                          (PRICE_ID, PRICE_DATE, MARKET_NAME, ITEM_NAME, TRADE_UNIT, GRADE, AVG_PRICE, UNIT_KG,
                           PRICE_PER_KG, GRADE_SCORE, CREATED_AT)
                          VALUES (SEQ_KAMIS_PRICE.NEXTVAL, TO_DATE(:1, 'YYYY-MM-DD'), '가락시장', '완숙토마토', :2, :3, :4, 1.0,
                                  :5, 4, SYSDATE) \
                          """
                    cursor.execute(sql, [d_str, unit_str, grade, raw_price, price_per_kg])
                    conn.commit()
                    print(f"📍 {d_str} 완료: {price_per_kg:,.0f}원/kg ({grade})")
                else:
                    print(f"⚪️ {d_str}: 토마토 데이터 없음")
            else:
                print(f"⚪️ {d_str}: 주말 또는 공휴일")

        except Exception as e:
            print(f"⚠️ {d_str} 오류: {e}")
            conn.rollback()

        time.sleep(0.2)  # API 부하 방지
        current_date += timedelta(days=1)

    print("\n✨ 모든 데이터 복구가 완료되었습니다! 이제 모델 학습을 시작하셔도 좋습니다.")
    cursor.close()
    conn.close()


if __name__ == "__main__":
    run_final_recovery()