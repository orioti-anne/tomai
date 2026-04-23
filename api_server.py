import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import oracledb
import platform
ORACLE_PATH = os.getenv("ORACLE_CLIENT_PATH")
if platform.system() == "Darwin":
    try:
        oracledb.init_oracle_client(lib_dir=ORACLE_PATH)
        print(f"[Oracle] thick mode: {ORACLE_PATH}")
    except Exception as e:
        print(f"[Oracle] {e}")

from smartfarm import create_app, db

app = create_app(enable_scheduler=True)

from flask import jsonify, request
from datetime import datetime

API_SECRET = os.getenv("API_SECRET", "tomai-internal-secret")

@app.before_request
def check_api_key():
    if request.path == "/health":
        return
    key = request.headers.get("X-API-Key")
    if key != API_SECRET:
        return jsonify({"error": "unauthorized"}), 401

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/monitoring/<int:cult_id>")
def api_monitoring(cult_id):
    try:
        from smartfarm.models import Cultivations, Farms
        from smartfarm.views.monitoring_views import (
            get_latest_daily_summary, get_previous_daily_summary,
            get_daily_chart_rows, get_latest_hourly_base_date,
            get_hourly_chart_rows, build_change_info, build_monitoring_logs
        )
        from smartfarm.services.weather_service import get_weather_alert_status

        latest_env = get_latest_daily_summary(cult_id)
        previous_env = get_previous_daily_summary(cult_id, latest_env.measure_date if latest_env else None)
        hourly_base_date = get_latest_hourly_base_date(cult_id)
        hourly_rows = get_hourly_chart_rows(cult_id, hourly_base_date)
        chart_rows = get_daily_chart_rows(cult_id, days=7)

        last_measured_at = None
        if hourly_rows:
            last_row = hourly_rows[-1]
            try:
                m_time = last_row.measure_time.time()
            except AttributeError:
                m_time = last_row.measure_time
            last_measured_at = datetime.combine(last_row.measure_date, m_time)

        cult = db.session.get(Cultivations, cult_id)
        current_alert = None
        if cult:
            farm = db.session.get(Farms, cult.farm_id)
            if farm:
                current_alert = get_weather_alert_status(farm.region_l1, farm.region_l2)

        logs = build_monitoring_logs(latest_env, weather_alert=current_alert, last_measured_at=last_measured_at)

        def env_to_dict(e):
            if not e:
                return None
            return {
                "daily_in_temp": float(e.daily_in_temp) if e.daily_in_temp is not None else None,
                "daily_in_humidity": float(e.daily_in_humidity) if e.daily_in_humidity is not None else None,
                "daily_in_co2": float(e.daily_in_co2) if e.daily_in_co2 is not None else None,
                "daily_acc_solar": float(e.daily_acc_solar) if e.daily_acc_solar is not None else None,
                "measure_date": e.measure_date.strftime("%Y-%m-%d") if e.measure_date else None,
            }

        return jsonify({
            "latest_env": env_to_dict(latest_env),
            "previous_env": env_to_dict(previous_env),
            "chart_labels": [r.measure_date.strftime("%m/%d") if r.measure_date else "" for r in chart_rows],
            "chart_temp_data": [float(r.daily_in_temp) if r.daily_in_temp is not None else None for r in chart_rows],
            "hourly_chart_labels": [r.measure_time.strftime("%H:%M") if r.measure_time else "" for r in hourly_rows],
            "hourly_chart_temp_data": [float(r.in_temp) if r.in_temp is not None else None for r in hourly_rows],
            "logs": logs,
            "temp_change": build_change_info(
                latest_env.daily_in_temp if latest_env else None,
                previous_env.daily_in_temp if previous_env else None, unit="°C"),
            "humidity_change": build_change_info(
                latest_env.daily_in_humidity if latest_env else None,
                previous_env.daily_in_humidity if previous_env else None, unit="%"),
            "co2_change": build_change_info(
                latest_env.daily_acc_solar if latest_env else None,
                previous_env.daily_acc_solar if previous_env else None, percent=True),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/growth/<int:cult_id>")
def api_growth(cult_id):
    try:
        from smartfarm.models import Cultivations
        from smartfarm.views.growth_views import (
            get_latest_environment, get_latest_growth_list,
            get_recent_env_7d_avg, recommend_environment, build_height_forecast
        )

        cult = db.session.get(Cultivations, cult_id)
        latest_env = get_latest_environment(cult_id)
        latest_growth_list = get_latest_growth_list(cult_id)
        latest_growth = latest_growth_list[0] if latest_growth_list else None
        env_avg_7d = get_recent_env_7d_avg(cult_id)
        recommended_env = recommend_environment(cult, latest_growth, latest_env)

        growth_forecasts = {}
        for gr in latest_growth_list:
            key = gr.plant_num if gr.plant_num is not None else 1
            growth_forecasts[key] = build_height_forecast(cult, gr, env_avg_7d)

        def growth_to_dict(gr):
            if not gr:
                return None
            return {
                "growth_id": gr.growth_id,
                "plant_num": gr.plant_num,
                "plant_height": float(gr.plant_height) if gr.plant_height is not None else None,
                "leaf_count": gr.leaf_count,
                "inspect_date": gr.inspect_date.strftime("%Y-%m-%d") if gr.inspect_date else None,
            }

        return jsonify({
            "recommended_env": recommended_env,
            "growth_forecasts": growth_forecasts,
            "latest_growth_list": [growth_to_dict(gr) for gr in latest_growth_list],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)
