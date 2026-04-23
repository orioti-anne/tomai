import psycopg2
import os
from flask import Blueprint, request, jsonify
from smartfarm import db
from smartfarm.models import PredictionDisplay

bp = Blueprint('display_api', __name__)

CLOUD_DB_URL = os.getenv("CLOUD_DB_URL")

def get_cloud_conn():
    return psycopg2.connect(CLOUD_DB_URL)

@bp.route('/api/receive-prediction', methods=['POST'])
def receive_prediction():
    data = request.get_json()
    if not data:
        return jsonify({"status": "fail", "message": "No data"}), 400
    try:
        new_entry = PredictionDisplay(
            category=data.get('type'),
            result_value=data.get('value'),
            target_date=data.get('target_date'),
            raw_json=str(data)
        )
        db.session.add(new_entry)
        db.session.commit()
        print(f"[GCP 수신완료] {data.get('type')}: {data.get('value')}")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "fail", "error": str(e)}), 500

@bp.route('/api/sync/farms', methods=['POST'])
def sync_farms():
    data = request.get_json()
    if not data:
        return jsonify({"status": "fail"}), 400
    try:
        conn = get_cloud_conn()
        cur = conn.cursor()
        for f in data.get('farms', []):
            cur.execute("""
                INSERT INTO cache_farms (farm_id, user_id, farm_name, region_l1, region_l2, total_area, is_active, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (farm_id) DO UPDATE SET
                    farm_name=EXCLUDED.farm_name,
                    region_l1=EXCLUDED.region_l1,
                    region_l2=EXCLUDED.region_l2,
                    total_area=EXCLUDED.total_area,
                    updated_at=NOW()
            """, (f['farm_id'], f['user_id'], f['farm_name'], f['region_l1'], f['region_l2'], f.get('total_area')))
        for c in data.get('cultivations', []):
            cur.execute("""
                INSERT INTO cache_cultivations (cult_id, farm_id, cult_name, item, item_variety, crop_cycle, planting_date, planting_area, status, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (cult_id) DO UPDATE SET
                    cult_name=EXCLUDED.cult_name,
                    item=EXCLUDED.item,
                    status=EXCLUDED.status,
                    updated_at=NOW()
            """, (c['cult_id'], c['farm_id'], c['cult_name'], c['item'], c.get('item_variety'),
                  c.get('crop_cycle'), c.get('planting_date'), c.get('planting_area'), c['status']))
        conn.commit()
        cur.close()
        conn.close()
        print(f"[SYNC] 농장/재배 정보 동기화 완료")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"[SYNC] 오류: {e}")
        return jsonify({"status": "fail", "error": str(e)}), 500

@bp.route('/api/sync/env', methods=['POST'])
def sync_env():
    data = request.get_json()
    if not data:
        return jsonify({"status": "fail"}), 400
    try:
        conn = get_cloud_conn()
        cur = conn.cursor()
        for e in data.get('env_list', []):
            cur.execute("""
                INSERT INTO cache_env_summary (cult_id, measure_date, daily_in_temp, daily_in_humidity, daily_in_co2, daily_acc_solar, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (cult_id) DO UPDATE SET
                    measure_date=EXCLUDED.measure_date,
                    daily_in_temp=EXCLUDED.daily_in_temp,
                    daily_in_humidity=EXCLUDED.daily_in_humidity,
                    daily_in_co2=EXCLUDED.daily_in_co2,
                    daily_acc_solar=EXCLUDED.daily_acc_solar,
                    updated_at=NOW()
            """, (e['cult_id'], e['measure_date'], e.get('daily_in_temp'),
                  e.get('daily_in_humidity'), e.get('daily_in_co2'), e.get('daily_acc_solar')))
        conn.commit()
        cur.close()
        conn.close()
        print(f"[SYNC] 환경 요약 동기화 완료")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"[SYNC] 오류: {e}")
        return jsonify({"status": "fail", "error": str(e)}), 500

@bp.route('/api/sync/prediction', methods=['POST'])
def sync_prediction():
    data = request.get_json()
    if not data:
        return jsonify({"status": "fail"}), 400
    try:
        conn = get_cloud_conn()
        cur = conn.cursor()
        for p in data.get('predictions', []):
            cur.execute("""
                INSERT INTO cache_prediction (cult_id, expected_harvest_date, expected_quantity, expected_sales, expected_price_per_kg, latest_market_price, prediction_date, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (cult_id) DO UPDATE SET
                    expected_harvest_date=EXCLUDED.expected_harvest_date,
                    expected_quantity=EXCLUDED.expected_quantity,
                    expected_sales=EXCLUDED.expected_sales,
                    expected_price_per_kg=EXCLUDED.expected_price_per_kg,
                    latest_market_price=EXCLUDED.latest_market_price,
                    prediction_date=EXCLUDED.prediction_date,
                    updated_at=NOW()
            """, (p['cult_id'], p.get('expected_harvest_date'), p.get('expected_quantity'),
                  p.get('expected_sales'), p.get('expected_price_per_kg'),
                  p.get('latest_market_price'), p.get('prediction_date')))
        conn.commit()
        cur.close()
        conn.close()
        print(f"[SYNC] 예측 결과 동기화 완료")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"[SYNC] 오류: {e}")
        return jsonify({"status": "fail", "error": str(e)}), 500
