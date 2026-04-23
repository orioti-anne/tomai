import os
import requests
from flask import Blueprint, render_template, request, g, redirect, url_for
from smartfarm.models import Cultivations, Farms
from smartfarm import db
bp = Blueprint("monitoring", __name__, url_prefix="/monitoring")
MAC_API_URL = os.getenv("MAC_API_URL", "http://100.126.59.34:8000")
MAC_API_KEY = os.getenv("MAC_API_KEY", "tomai-internal-secret")
def get_user_cultivations(user_id):
    return (
        Cultivations.query
        .join(Farms, Cultivations.farm_id == Farms.farm_id)
        .filter(Farms.user_id == user_id, Farms.is_active == 'Y')
        .filter(Cultivations.status != 'hidden')
        .all()
    )
def get_selected_cultivation(cultivations, cult_id):
    if not cultivations:
        return None
    if cult_id is not None:
        for cult in cultivations:
            if cult.cult_id == cult_id:
                return cult
    return cultivations[0]
def build_monitoring_logs(latest_env=None, weather_alert=None, last_measured_at=None):
    return []
def fetch_monitoring_data(cult_id):
    try:
        res = requests.get(
            f"{MAC_API_URL}/api/monitoring/{cult_id}",
            headers={"X-API-Key": MAC_API_KEY},
            timeout=10
        )
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"[monitoring] API 호출 실패: {e}")
    return {}
@bp.route("/")
def monitoring():
    if not g.user:
        return redirect(url_for("auth.login"))
    cultivations = get_user_cultivations(g.user.user_id)
    selected_cult_id = request.args.get("cult_id", type=int)
    selected_cultivation = get_selected_cultivation(cultivations, selected_cult_id)
    data = {}
    if selected_cultivation:
        data = fetch_monitoring_data(selected_cultivation.cult_id)
    return render_template(
        "monitoring.html",
        cultivations=cultivations,
        selected_cultivation=selected_cultivation,
        latest_env=data.get("latest_env"),
        previous_env=data.get("previous_env"),
        chart_labels=data.get("chart_labels", []),
        chart_temp_data=data.get("chart_temp_data", []),
        hourly_chart_labels=data.get("hourly_chart_labels", []),
        hourly_chart_temp_data=data.get("hourly_chart_temp_data", []),
        hourly_base_date=data.get("hourly_base_date"),
        hourly_data_count=len([v for v in data.get("hourly_chart_temp_data", []) if v is not None]),
        logs=data.get("logs", []),
        temp_change=data.get("temp_change"),
        humidity_change=data.get("humidity_change"),
        co2_change=data.get("co2_change"),
        solar_change=data.get("solar_change"),
    )
