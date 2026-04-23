import os
import psycopg2
from datetime import datetime

CLOUD_DB_URL = os.getenv("CLOUD_DB_URL")

def get_cloud_conn():
    return psycopg2.connect(CLOUD_DB_URL)

def sync_farms_and_cultivations(app):
    """농장/재배 정보를 클라우드 DB에 동기화"""
    from smartfarm.models import Farms, Cultivations
    with app.app_context():
        try:
            conn = get_cloud_conn()
            cur = conn.cursor()

            farms = Farms.query.filter_by(is_active='Y').all()
            for f in farms:
                cur.execute("""
                    INSERT INTO cache_farms (farm_id, user_id, farm_name, region_l1, region_l2, total_area, is_active, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (farm_id) DO UPDATE SET
                        farm_name=EXCLUDED.farm_name,
                        region_l1=EXCLUDED.region_l1,
                        region_l2=EXCLUDED.region_l2,
                        total_area=EXCLUDED.total_area,
                        updated_at=EXCLUDED.updated_at
                """, (f.farm_id, f.user_id, f.farm_name, f.region_l1, f.region_l2,
                      float(f.total_area) if f.total_area else None, f.is_active, datetime.now()))

            cultivations = Cultivations.query.filter(Cultivations.status != 'hidden').all()
            for c in cultivations:
                cur.execute("""
                    INSERT INTO cache_cultivations (cult_id, farm_id, cult_name, item, item_variety, crop_cycle, planting_date, planting_area, status, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cult_id) DO UPDATE SET
                        cult_name=EXCLUDED.cult_name,
                        item=EXCLUDED.item,
                        status=EXCLUDED.status,
                        updated_at=EXCLUDED.updated_at
                """, (c.cult_id, c.farm_id, c.cult_name, c.item, c.item_variety,
                      int(c.crop_cycle) if c.crop_cycle else None,
                      c.planting_date, float(c.planting_area) if c.planting_area else None,
                      c.status, datetime.now()))

            conn.commit()
            cur.close()
            conn.close()
            print(f"[SYNC] 농장/재배 정보 동기화 완료: {len(farms)}개 농장, {len(cultivations)}개 재배")
        except Exception as e:
            print(f"[SYNC] 농장/재배 동기화 오류: {e}")

def sync_env_summary(app):
    """최신 환경 요약을 클라우드 DB에 동기화"""
    from smartfarm.models import EnvSummary, Cultivations
    with app.app_context():
        try:
            conn = get_cloud_conn()
            cur = conn.cursor()

            cults = Cultivations.query.filter(Cultivations.status != 'hidden').all()
            for c in cults:
                latest = EnvSummary.query.filter_by(cult_id=c.cult_id).order_by(EnvSummary.measure_date.desc()).first()
                if not latest:
                    continue
                cur.execute("""
                    INSERT INTO cache_env_summary (cult_id, measure_date, daily_in_temp, daily_in_humidity, daily_in_co2, daily_acc_solar, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cult_id) DO UPDATE SET
                        measure_date=EXCLUDED.measure_date,
                        daily_in_temp=EXCLUDED.daily_in_temp,
                        daily_in_humidity=EXCLUDED.daily_in_humidity,
                        daily_in_co2=EXCLUDED.daily_in_co2,
                        daily_acc_solar=EXCLUDED.daily_acc_solar,
                        updated_at=EXCLUDED.updated_at
                """, (c.cult_id, latest.measure_date,
                      float(latest.daily_in_temp) if latest.daily_in_temp else None,
                      float(latest.daily_in_humidity) if latest.daily_in_humidity else None,
                      float(latest.daily_in_co2) if latest.daily_in_co2 else None,
                      float(latest.daily_acc_solar) if latest.daily_acc_solar else None,
                      datetime.now()))

            conn.commit()
            cur.close()
            conn.close()
            print(f"[SYNC] 환경 요약 동기화 완료")
        except Exception as e:
            print(f"[SYNC] 환경 요약 동기화 오류: {e}")

def sync_predictions(app):
    """최신 예측 결과를 클라우드 DB에 동기화"""
    from smartfarm.models import PredictionResults, Cultivations
    with app.app_context():
        try:
            conn = get_cloud_conn()
            cur = conn.cursor()

            cults = Cultivations.query.filter(Cultivations.status != 'hidden').all()
            for c in cults:
                pred = PredictionResults.query.filter_by(cult_id=c.cult_id).order_by(PredictionResults.prediction_date.desc()).first()
                if not pred:
                    continue
                cur.execute("""
                    INSERT INTO cache_prediction (cult_id, expected_harvest_date, expected_quantity, expected_sales, expected_price_per_kg, latest_market_price, prediction_date, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cult_id) DO UPDATE SET
                        expected_harvest_date=EXCLUDED.expected_harvest_date,
                        expected_quantity=EXCLUDED.expected_quantity,
                        expected_sales=EXCLUDED.expected_sales,
                        expected_price_per_kg=EXCLUDED.expected_price_per_kg,
                        latest_market_price=EXCLUDED.latest_market_price,
                        prediction_date=EXCLUDED.prediction_date,
                        updated_at=EXCLUDED.updated_at
                """, (c.cult_id, pred.expected_harvest_date,
                      float(pred.expected_quantity) if pred.expected_quantity else None,
                      float(pred.expected_sales) if pred.expected_sales else None,
                      float(pred.expected_price_per_kg) if pred.expected_price_per_kg else None,
                      float(pred.latest_market_price) if pred.latest_market_price else None,
                      pred.prediction_date, datetime.now()))

            conn.commit()
            cur.close()
            conn.close()
            print(f"[SYNC] 예측 결과 동기화 완료")
        except Exception as e:
            print(f"[SYNC] 예측 결과 동기화 오류: {e}")

def run_full_sync(app):
    """전체 동기화 실행"""
    print(f"[SYNC] 전체 동기화 시작: {datetime.now()}")
    sync_farms_and_cultivations(app)
    sync_env_summary(app)
    sync_predictions(app)
    print(f"[SYNC] 전체 동기화 완료: {datetime.now()}")
