from flask import Blueprint, render_template, request, g, redirect, url_for
from sqlalchemy import func
from datetime import datetime
from smartfarm.models import Cultivations, Farms, EnvSummary, EnvCleaned
from smartfarm.services.weather_service import get_weather_alert_status

bp = Blueprint("monitoring", __name__, url_prefix="/monitoring")


def get_user_cultivations(user_id):
    return (
        Cultivations.query
        .join(Farms, Cultivations.farm_id == Farms.farm_id)
        .filter(
            Farms.user_id == user_id,
            Farms.is_active == 'Y'
        )
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

def get_latest_daily_summary(cult_id):
    return (
        EnvSummary.query
        .filter(EnvSummary.cult_id == cult_id)
        .order_by(EnvSummary.measure_date.desc())
        .first()
    )

def get_previous_daily_summary(cult_id, latest_date):
    if not latest_date:
        return None

    return (
        EnvSummary.query
        .filter(
            EnvSummary.cult_id == cult_id,
            EnvSummary.measure_date < latest_date
        )
        .order_by(EnvSummary.measure_date.desc())
        .first()
    )

def get_daily_chart_rows(cult_id, days=7):
    rows = (
        EnvSummary.query
        .filter(EnvSummary.cult_id == cult_id)
        .order_by(EnvSummary.measure_date.desc())
        .limit(days)
        .all()
    )
    return list(reversed(rows))

def get_latest_hourly_base_date(cult_id):
    return (
        EnvCleaned.query
        .with_entities(func.max(EnvCleaned.measure_date))
        .filter(EnvCleaned.cult_id == cult_id)
        .scalar()
    )

def get_hourly_chart_rows(cult_id, base_date):
    if not base_date:
        return []

    return (
        EnvCleaned.query
        .filter(
            EnvCleaned.cult_id == cult_id,
            EnvCleaned.measure_date == base_date
        )
        .order_by(EnvCleaned.measure_time.asc())
        .all()
    )

def build_change_info(current_value, previous_value, unit="", percent=False):
    if current_value is None or previous_value is None:
        return None

    current_value = float(current_value)
    previous_value = float(previous_value)
    diff = current_value - previous_value

    if percent:
        if previous_value == 0:
            return None

        rate = (diff / previous_value) * 100
        return {
            "direction": "up" if rate > 0 else "down" if rate < 0 else "same",
            "value": abs(rate),
            "text": f"{abs(rate):.1f}% 어제 대비",
        }

    return {
        "direction": "up" if diff > 0 else "down" if diff < 0 else "same",
        "value": abs(diff),
        "text": f"{abs(diff):.1f}{unit} 어제 대비",
    }

def build_monitoring_logs(latest_env, weather_alert=None, last_measured_at=None):
    logs = []

    if weather_alert:
        logs.append({
            "level": "danger",
            "title": f"🚨 {weather_alert.get('title', '기상 특보 발령')}",
            "message": weather_alert.get('message', '기상 상황을 확인하고 시설물을 점검하세요.'),
            "time_text": "실시간",
        })

    if not latest_env:
        return logs

    date_text = latest_env.measure_date.strftime("%m/%d") if latest_env.measure_date else "-"


    if latest_env.daily_in_temp is not None:
        if latest_env.daily_in_temp >= 28:
            logs.append({
                "level": "danger",
                "title": "🌡️ 고온 주의",
                "message": f"평균 온도가 {latest_env.daily_in_temp:.1f}°C로 높습니다. 환기창 개방 및 차광막 가동을 권장합니다.",
                "time_text": date_text,
            })
        elif latest_env.daily_in_temp <= 12:
            logs.append({
                "level": "warning",
                "title": "❄️ 저온 주의",
                "message": f"야간 온도 하락에 대비하여 보온 커튼 점검이 필요합니다. (현재 {latest_env.daily_in_temp:.1f}°C)",
                "time_text": date_text,
            })
        else:
            logs.append({
                "level": "success",
                "title": "온도 최적 상태",
                "message": f"실내 온도({latest_env.daily_in_temp:.1f}°C)가 생육에 적합한 범위 내에 있습니다.",
                "time_text": date_text,
            })

    if latest_env.daily_in_humidity is not None:
        if 55 <= latest_env.daily_in_humidity <= 75:
            logs.append({
                "level": "success",
                "title": "습도 적정",
                "message": f"실내 습도({latest_env.daily_in_humidity:.1f}%)가 안정적입니다.",
                "time_text": date_text,
            })
        else:
            logs.append({
                "level": "warning",
                "title": "💧 습도 관리 필요",
                "message": "습도가 적정 범위를 벗어났습니다. 제습 또는 관수량 조절이 필요합니다.",
                "time_text": date_text,
            })


    full_time_text = last_measured_at.strftime("%m/%d %H:%M") if last_measured_at else date_text

    logs.append({
        "level": "info",
        "title": "시스템 알림",
        "message": "환경센서가 최신 데이터를 성공적으로 수집하였습니다.",
        "time_text": full_time_text
    })

    return logs[:10]


@bp.route("/")
def monitoring():
    if not g.user:
        return redirect(url_for("auth.login"))

    cultivations = get_user_cultivations(g.user.user_id)
    selected_cult_id = request.args.get("cult_id", type=int)
    selected_cultivation = get_selected_cultivation(cultivations, selected_cult_id)

    latest_env = None
    previous_env = None
    logs = []

    chart_labels = []
    chart_temp_data = []

    hourly_chart_labels = []
    hourly_chart_temp_data = []
    hourly_base_date = None
    hourly_data_count = 0

    temp_change = None
    humidity_change = None
    co2_change = None
    solar_change = None

    if selected_cultivation:
        cult_id = selected_cultivation.cult_id
        latest_env = get_latest_daily_summary(cult_id)

        hourly_base_date = get_latest_hourly_base_date(cult_id)
        hourly_rows = get_hourly_chart_rows(cult_id, hourly_base_date)

        last_measured_at = None
        if hourly_rows:
            last_row = hourly_rows[-1]
            if last_row.measure_date and last_row.measure_time:
                try:
                    m_time = last_row.measure_time.time()
                except AttributeError:
                    m_time = last_row.measure_time

                last_measured_at = datetime.combine(last_row.measure_date, m_time)

        if latest_env:
            previous_env = get_previous_daily_summary(cult_id, latest_env.measure_date)

            temp_change = build_change_info(
                latest_env.daily_in_temp,
                previous_env.daily_in_temp if previous_env else None,
                unit="°C",
            )
            humidity_change = build_change_info(
                latest_env.daily_in_humidity,
                previous_env.daily_in_humidity if previous_env else None,
                unit="%",
            )
            co2_change = build_change_info(
                latest_env.daily_in_co2,
                previous_env.daily_in_co2 if previous_env else None,
                unit=" ppm",
            )
            solar_change = build_change_info(
                latest_env.daily_acc_solar,
                previous_env.daily_acc_solar if previous_env else None,
                percent=True,
            )

            # 1. 농장 정보를 가져와서 지역명(l1, l2) 추출
            farm = Farms.query.get(selected_cultivation.farm_id)
            current_alert = None

            if farm:
                current_alert = get_weather_alert_status(farm.region_l1, farm.region_l2)

            logs = build_monitoring_logs(
                latest_env,
                weather_alert=current_alert,
                last_measured_at=last_measured_at
            )

        chart_rows = get_daily_chart_rows(cult_id, days=7)
        chart_labels = [row.measure_date.strftime("%m/%d") if row.measure_date else "" for row in chart_rows]
        chart_temp_data = [float(row.daily_in_temp) if row.daily_in_temp is not None else None for row in chart_rows]

        hourly_chart_labels = [row.measure_time.strftime("%H:%M") if row.measure_time else "" for row in hourly_rows]
        hourly_chart_temp_data = [float(row.in_temp) if row.in_temp is not None else None for row in hourly_rows]
        hourly_data_count = len([v for v in hourly_chart_temp_data if v is not None])

    return render_template(
        "monitoring.html",
        cultivations=cultivations,
        selected_cultivation=selected_cultivation,
        latest_env=latest_env,
        previous_env=previous_env,
        chart_labels=chart_labels,
        chart_temp_data=chart_temp_data,
        hourly_chart_labels=hourly_chart_labels,
        hourly_chart_temp_data=hourly_chart_temp_data,
        hourly_base_date=hourly_base_date,
        hourly_data_count=hourly_data_count,
        logs=logs,
        temp_change=temp_change,
        humidity_change=humidity_change,
        co2_change=co2_change,
        solar_change=solar_change,
    )