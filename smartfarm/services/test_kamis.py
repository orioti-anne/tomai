import requests
import json
from datetime import datetime, timedelta

# API 설정
API_KEY = "d9c4bd57-6f86-417e-85f0-4db5254fe40e"
USER_ID = "7367"
BASE_URL = "http://www.kamis.or.kr/service/price/xml.do"


def check_only_ripe_tomato(target_date):
    print(f"\n🔍 [{target_date}] 완숙토마토 필터링 테스트")

    params = {
        'action': 'dailyPriceByCategoryList',
        'p_product_cls_code': '02',  # 도매
        'p_item_category_code': '200',  # 채소류
        'p_item_code': '225',  # 토마토 품목
        'p_kind_code': '11',  # 완숙 품종 설정 (API가 무시할 때를 대비해 로직으로 재필터링)
        'p_country_code': '1101',  # 가락시장
        'p_regday': target_date,
        'p_convert_kg_yn': 'N',  # 원본 가격 확인을 위해 N
        'p_cert_key': API_KEY,
        'p_cert_id': USER_ID,
        'p_returntype': 'json'
    }

    try:
        res = requests.get(BASE_URL, params=params, timeout=10)
        data = res.json()

        if data.get('data') and isinstance(data['data'].get('item'), list):
            all_items = data['data']['item']

            # [핵심 로직] 배추, 방울, 대추를 제외하고 '토마토'만 남기기
            ripe_tomatoes = [
                item for item in all_items
                if ('토마토' in item.get('item_name', '') or '토마토' in item.get('kind_name', ''))
                   and '방울' not in item.get('kind_name', '')
                   and '대추' not in item.get('kind_name', '')
                   and '포기' not in item.get('unit', '')  # 배추 방어
            ]

            if ripe_tomatoes:
                print(f"✅ 조건에 맞는 데이터 {len(ripe_tomatoes)}건 발견")
                for i, tomato in enumerate(ripe_tomatoes):
                    kind = tomato.get('kind_name')
                    rank = tomato.get('rank')
                    unit = tomato.get('unit')
                    price_str = tomato.get('dpr1', '0').replace(',', '')

                    if price_str == '-':  # 데이터가 없는 경우
                        continue

                    price = float(price_str)

                    # 1kg 환산가 계산 (예: 5kg 상자 가격 / 5)
                    calc_kg_price = 0
                    if '5kg' in unit:
                        calc_kg_price = price / 5
                    elif '10kg' in unit:
                        calc_kg_price = price / 10
                    elif '1kg' in unit:
                        calc_kg_price = price

                    print(
                        f"   [{i + 1}] {kind} ({rank}) | 단위: {unit} | 원본가: {price:,.0f}원 | 1kg환산: {calc_kg_price:,.0f}원")
            else:
                print("❌ 검색 조건(완숙토마토)에 일치하는 데이터가 리스트에 없습니다.")
        else:
            print("⚪ 해당 날짜는 시장 휴무일이거나 데이터가 응답되지 않았습니다.")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    # 최근 며칠간의 데이터를 샘플링해서 확인
    sample_dates = [
        "2026-04-02",
        "2026-04-01",
        "2026-03-31"
    ]

    for date in sample_dates:
        check_only_ripe_tomato(date)