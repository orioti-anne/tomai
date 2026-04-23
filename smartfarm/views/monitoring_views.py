import os
from datetime import datetime
from flask import Blueprint, render_template, g, redirect, url_for, request
from smartfarm.models import Cultivations, EnvSummary, EnvCleaned, Farms
from smartfarm.services.weather_service import get_weather_alert_status

bp = Blueprint("monitoring", __name__, url_prefix="/monitoring")

def get_user_cultivations(user_id):
    return (Cultivations.query
            .join(Farms, Cultivations.farm_id == Farms.farm_id)
            .filter(Farms.user_id == user_id, Farms.is_active == 'Y',
                    Cultivations.status == 'active')
            .all())

def get_selected_cultivation(cultivations, cult_id):
    if cult_id:
        return next((c for c in cultivations if c.cult_id == cult_id), None)
    return cultivations[0] if cultivations else None

def build_monitoring_logs(latest_env=None, weather_alert=None, last_measured_at=None):
    logs = []
    if weather_alert:
        logs.append({"level": "danger",
                     "title": f"🚨 {weather_alert.get('title', '기상 특보 발령')}",
                     "message": weather_alert.get('message', ''),
                     "time_text": "실시간"})
    if not latest_env:
        return logs
    if isinstance(latest_env, dict):
        date_text = (latest_env.get("measure_date") or "-")[5:10].replace("-", "/")
        temp = latest_env.get("daily_in_temp")
        humidity = latest_env.get("daily_in_humidity")
    else:
        date_text = latest_env.measure_date.strftime("%m/%d") if latest_env.measure_date else "-"
        temp = latest_env.daily_in_temp
        humidity = latest_env.daily_in_humidity

    if temp is not None:
        temp = float(temp)
        if temp >= 28:
            logs.append({"level": "danger", "title": "🌡️ 고온 주의",
                         "message": f"평균 온도가 {temp:.1f}°C로 높습니다.", "time_text": date_text})
        elif temp <= 12:
            logs.append({"level": "warning", "title": "❄️ 저온 주의",
                         "message": f"야간 온도 하락에 대비하세요. (현재 {temp:.1f}°C)", "time_text": date_text})
        else:
            logs.append({"level": "success", "title": "온도 최적 상태",
                         "message": f"실내 온도({temp:.1f}°C)가 생육에 적합한 범위 내에 있습니다.",
                         "time_text": date_text})
    if humidity is not None:
        humidity = float(humidity)
        if 55 <= humidity <= 75:
            logs.append({"level": "success", "title": "습도 적정",
                         "message": f"실내 습도({humidity:.1f}%)가 안정적입니다.", "time_text": date_text})
        else:
            logs.append({"level": "warning", "title": "💧 습도 관리 필요",
                         "message": "습도가 적정 범위를 벗어났습니다.", "time_text": date_text})
    full_time_text = last_measured_at.strftime("%m/%d %H:%M") if last_measured_at else date_text
    logs.append({"level": "info", "title": "시스템 알림",
                 "message": "환경센서가 최신 데이터를 성공적으로 수집하였습니다.",
                 "time_text": full_time_text})
    return logs[:10]

def change_info(curr_val, prev_val, unit="", percent=False):
    if curr_val is None or prev_val is None:
        return {"direction": "same", "value": 0, "text": "데이터 없음"}
    diff = float(curr_val) - float(prev_val)
    direction = "up" if diff > 0 else ("down" if diff < 0 else "same")
    text = f"{abs(diff):.1f}{'%' if percent else unit} 어제 대비"
    return {"direction": direction, "value": abs(diff), "text": text}

def fetch_monitoring_data(cult_id):
    latest = EnvSummary.query.filter_by(cult_id=cult_id).order_by(EnvSummary.measure_date.desc()).first()
    previous = None
    if latest:
        previous = EnvSummary.query.filter(
            EnvSummary.cult_id == cult_id,
            EnvSummary.measure_date < latest.measure_date
        ).order_by(EnvSummary.measure_date.desc()).first()

    chart_rows = list(reversed(
        EnvSummary.query.filter_by(cult_id=cult_id)
        .order_by(EnvSummary.measure_date.desc()).limit(7).all()
    ))

    today = datetime.now().date()
    hourly_rows = EnvCleaned.query.filter(
        EnvCleaned.cult_id == cult_id,
        EnvCleaned.measure_date == today
    ).order_by(EnvCleaned.measure_hour.asc()).all()

    def env_dict(e):
        if not e:
            return None
        return {
            "measure_date": str(e.measure_date),
            "daily_in_temp": float(e.daily_in_temp) if e.daily_in_temp else None,
            "daily_in_humidity": float(e.daily_in_humidity) if e.daily_in_humidity else None,
            "daily_in_co2": float(e.daily_in_co2) if e.daily_in_co2 else None,
            "daily_acc_solar": float(e.daily_acc_solar) if e.daily_acc_solar else None,
        }

    return {
        "latest_env": env_dict(latest),
        "previous_env": env_dict(previous),
        "chart_labels": [r.measure_date.strftime("%m/%d") for r in chart_rows],
        "chart_temp_data": [float(r.daily_in_temp) if r.daily_in_temp else None for r in chart_rows],
        "hourly_chart_labels": [f"{r.measure_hour:02d}:00" for r in hourly_rows],
        "hourly_chart_temp_data": [float(r.in_temp) if r.in_temp else None for r in hourly_rows],
        "hourly_base_date": str(today) if hourly_rows else None,
        "temp_change": change_info(latest.daily_in_temp if latest else None,
                                   previous.daily_in_temp if previous else None, "°C"),
        "humidity_change": change_info(latest.daily_in_humidity if latest else None,
                                       previous.daily_in_humidity if previous else None, "%"),
        "co2_change": change_info(latest.daily_in_co2 if latest else None,
                                  previous.daily_in_co2 if previous else None, " ppm"),
        "solar_change": change_info(latest.daily_acc_solar if latest else None,
                                    previous.daily_acc_solar if previous else None, "", percent=True),
    }

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
        farm = Farms.query.get(selected_cultivation.farm_id)
        weather_alert = get_weather_alert_status(farm.region_l1, farm.region_l2) if farm else None
        data["logs"] = build_monitoring_logs(data.get("latest_env"), weather_alert=weather_alert)

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
        temp_change=data.get("temp_change"),
        humidity_change=data.get("humidity_change"),
        co2_change=data.get("co2_change"),
        solar_change=data.get("solar_change"),
        logs=data.get("logs", []),
    )
