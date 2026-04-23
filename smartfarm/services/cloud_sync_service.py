import os
import requests
from datetime import datetime

GCP_SYNC_URL = os.getenv("GCP_SYNC_URL", "http://136.119.67.188:5000")

def sync_farms_and_cultivations(app):
    from smartfarm.models import Farms, Cultivations
    with app.app_context():
        try:
            farms = Farms.query.filter_by(is_active='Y').all()
            cultivations = Cultivations.query.filter(Cultivations.status != 'hidden').all()

            payload = {
                "farms": [{
                    "farm_id": f.farm_id,
                    "user_id": f.user_id,
                    "farm_name": f.farm_name,
                    "region_l1": f.region_l1,
                    "region_l2": f.region_l2,
                    "total_area": float(f.total_area) if f.total_area else None,
                    "is_active": f.is_active
                } for f in farms],
                "cultivations": [{
                    "cult_id": c.cult_id,
                    "farm_id": c.farm_id,
                    "cult_name": c.cult_name,
                    "item": c.item,
                    "item_variety": c.item_variety,
                    "crop_cycle": int(c.crop_cycle) if c.crop_cycle else None,
                    "planting_date": c.planting_date.strftime("%Y-%m-%d") if c.planting_date else None,
                    "planting_area": float(c.planting_area) if c.planting_area else None,
                    "status": c.status
                } for c in cultivations]
            }

            res = requests.post(f"{GCP_SYNC_URL}/api/sync/farms", json=payload, timeout=10)
            print(f"[SYNC] 농장/재배 동기화: {res.status_code}")
        except Exception as e:
            print(f"[SYNC] 농장/재배 오류: {e}")

def sync_env_summary(app):
    from smartfarm.models import Cultivations, EnvSummary
    with app.app_context():
        try:
            cults = Cultivations.query.filter(Cultivations.status != 'hidden').all()
            env_list = []
            for c in cults:
                latest = EnvSummary.query.filter_by(cult_id=c.cult_id).order_by(EnvSummary.measure_date.desc()).first()
                if not latest:
                    continue
                env_list.append({
                    "cult_id": c.cult_id,
                    "measure_date": latest.measure_date.strftime("%Y-%m-%d") if latest.measure_date else None,
                    "daily_in_temp": float(latest.daily_in_temp) if latest.daily_in_temp else None,
                    "daily_in_humidity": float(latest.daily_in_humidity) if latest.daily_in_humidity else None,
                    "daily_in_co2": float(latest.daily_in_co2) if latest.daily_in_co2 else None,
                    "daily_acc_solar": float(latest.daily_acc_solar) if latest.daily_acc_solar else None,
                })

            res = requests.post(f"{GCP_SYNC_URL}/api/sync/env", json={"env_list": env_list}, timeout=10)
            print(f"[SYNC] 환경 요약 동기화: {res.status_code}")
        except Exception as e:
            print(f"[SYNC] 환경 요약 오류: {e}")

def sync_predictions(app):
    from smartfarm.models import Cultivations, PredictionResults
    with app.app_context():
        try:
            cults = Cultivations.query.filter(Cultivations.status != 'hidden').all()
            predictions = []
            for c in cults:
                pred = PredictionResults.query.filter_by(cult_id=c.cult_id).order_by(PredictionResults.prediction_date.desc()).first()
                if not pred:
                    continue
                predictions.append({
                    "cult_id": c.cult_id,
                    "expected_harvest_date": pred.expected_harvest_date.strftime("%Y-%m-%d") if pred.expected_harvest_date else None,
                    "expected_quantity": float(pred.expected_quantity) if pred.expected_quantity else None,
                    "expected_sales": float(pred.expected_sales) if pred.expected_sales else None,
                    "expected_price_per_kg": float(pred.expected_price_per_kg) if pred.expected_price_per_kg else None,
                    "latest_market_price": float(pred.latest_market_price) if pred.latest_market_price else None,
                    "prediction_date": pred.prediction_date.strftime("%Y-%m-%d %H:%M:%S") if pred.prediction_date else None,
                })

            res = requests.post(f"{GCP_SYNC_URL}/api/sync/prediction", json={"predictions": predictions}, timeout=10)
            print(f"[SYNC] 예측 결과 동기화: {res.status_code}")
        except Exception as e:
            print(f"[SYNC] 예측 결과 오류: {e}")

def run_full_sync(app):
    print(f"[SYNC] 전체 동기화 시작: {datetime.now()}")
    sync_farms_and_cultivations(app)
    sync_env_summary(app)
    sync_predictions(app)
    print(f"[SYNC] 전체 동기화 완료: {datetime.now()}")
