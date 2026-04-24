import os
import httpx
import datetime
import urllib3
from datetime import date
from dotenv import load_dotenv
from sqlalchemy import text
from smartfarm import db

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()


class PriceCollector:
    API_KEY = os.getenv("KAMIS_API_KEY")
    USER_ID = os.getenv("KAMIS_USER_ID")
    BASE_URL = os.getenv("KAMIS_BASE_URL", "http://www.kamis.or.kr/service/price/xml.do")

    @staticmethod
    def collect_tomato_price():

        if not PriceCollector.API_KEY or not PriceCollector.USER_ID:
            print("❌ [ERROR] KAMIS_API_KEY 또는 KAMIS_USER_ID가 .env에 설정되지 않았습니다.")
            return {"success": False, "message": "API 인증 정보 누락"}

        for i in range(8):
            target_date = date.today() - datetime.timedelta(days=i)
            target_str = target_date.strftime("%Y-%m-%d")

            params = {
                'action': 'dailyPriceByCategoryList',
                'p_product_cls_code': '02',  # 도매
                'p_item_category_code': '200',  # 채소류
                'p_item_code': '225',  # 토마토 품목 코드
                'p_kind_code': '11',  # 완숙 품종 코드
                'p_country_code': '1101',  # 가락시장 기준
                'p_regday': target_str,
                'p_convert_kg_yn': 'N',  # 수동 환산을 위해 N 설정
                'p_cert_key': PriceCollector.API_KEY,
                'p_cert_id': PriceCollector.USER_ID,
                'p_returntype': 'json'
            }

            try:
                import subprocess, json as _json, urllib.parse
                _url = PriceCollector.BASE_URL + "?" + urllib.parse.urlencode(params)
                _res = subprocess.run(
                    ["curl", "-k", "-s", "--max-time", "15", _url],
                    capture_output=True, text=True
                )
                try:
                    data = _json.loads(_res.stdout)
                except Exception:
                    print(f"{target_str}: JSON 응답이 아닙니다. (시장 휴무 또는 서버 오류)")
                    continue

                if data.get('data') and isinstance(data['data'].get('item'), list):
                    all_items = data['data']['item']

                    # 완숙토마토만 필터링
                    ripe_tomatoes = [
                        item for item in all_items
                        if ('토마토' in item.get('item_name', '') or '토마토' in item.get('kind_name', ''))
                           and '방울' not in item.get('kind_name', '')
                           and '대추' not in item.get('kind_name', '')
                           and '포기' not in item.get('unit', '')
                    ]

                    if ripe_tomatoes:
                        target_item = next((i for i in ripe_tomatoes if i.get('rank') == '상품'), ripe_tomatoes[0])

                        price_str = str(target_item.get('dpr1', '0')).replace(',', '')
                        if price_str == '-' or price_str == '0':
                            print(f"{target_str}: 가격 데이터가 '-' 입니다. 전날로 넘어갑니다.")
                            continue

                        raw_price = float(price_str)
                        unit_str = target_item.get('unit', '5kg').strip()
                        grade = target_item.get('rank', '상품').strip()

                        if '5kg' in unit_str:
                            price_per_kg = raw_price / 5
                        elif '10kg' in unit_str:
                            price_per_kg = raw_price / 10
                        elif '1kg' in unit_str:
                            price_per_kg = raw_price
                        else:
                            price_per_kg = raw_price / 5
                            print(f"⚠{target_str}: 특이 단위 발견({unit_str}), 5kg 기준으로 계산함")

                        # DB 저장 (PostgreSQL UPSERT)
                        query = text("""
                            INSERT INTO kamis_tomato_price 
                                (price_date, market_name, item_name, trade_unit, grade, 
                                 avg_price, unit_kg, price_per_kg, grade_score, created_at)
                            VALUES 
                                (:p_date::date, '가락시장', '완숙토마토', :p_unit, :p_grade,
                                 :p_raw_price, 1.0, :p_price_kg, 4, NOW())
                            ON CONFLICT (price_date, item_name, grade) 
                            DO UPDATE SET
                                avg_price = EXCLUDED.avg_price,
                                price_per_kg = EXCLUDED.price_per_kg,
                                trade_unit = EXCLUDED.trade_unit,
                                created_at = NOW()
                        """)

                        db.session.execute(query, {
                            "p_date": target_str,
                            "p_grade": grade,
                            "p_raw_price": raw_price,
                            "p_price_kg": price_per_kg,
                            "p_unit": unit_str
                        })
                        db.session.commit()

                        print(f"[수집성공] {target_str} | {grade} | {unit_str} | 1kg환산: {price_per_kg:,.0f}원")
                        return {"success": True, "details": f"{target_str} {grade}: {price_per_kg:,.0f}원/kg"}

                print(f"{target_str}: 유효한 완숙토마토 데이터 없음, 전날 시도 중...")

            except Exception as e:
                print(f"{target_str} 수집 중 시스템 오류 발생: {e}")
                db.session.rollback()
                continue

        return {"success": False, "message": "최근 7일 이내에 유효한 완숙토마토 데이터를 찾을 수 없습니다."}


if __name__ == "__main__":
    from smartfarm import create_app

    app = create_app()
    with app.app_context():
        result = PriceCollector.collect_tomato_price()
        print(f"최종 결과: {result}")