from flask import Blueprint, render_template, url_for, redirect, g, request
from datetime import date, timedelta
from sqlalchemy import func, text

from smartfarm import db
from smartfarm.models import Cultivations, PredictionResults, Farms, Growth, EnvSummary
from smartfarm.services.weather_service import get_weather, get_weather_alert_status
from smartfarm.views.monitoring_views import build_monitoring_logs

bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')


@bp.route('/')
def index():
    if not g.user:
        return redirect(url_for('auth.login'))

    cult_id = request.args.get("cult_id", type=int)

    cult_list = (
        Cultivations.query
        .join(Farms, Cultivations.farm_id == Farms.farm_id)
        .filter(
            Farms.user_id == g.user.user_id,
            Farms.is_active == 'Y',
            Cultivations.status == 'active'
        )
        .all()
    )

    selected_cult = Cultivations.query.get(cult_id) if cult_id else (cult_list[0] if cult_list else None)

    farm = None
    weather = None
    prediction = None
    latest_growth_list = []
    latest_env = None
    logs = []
    priority_log = None

    if selected_cult:
        farm = Farms.query.get(selected_cult.farm_id)

        prediction = (
            PredictionResults.query
            .filter_by(cult_id=selected_cult.cult_id)
            .order_by(PredictionResults.prediction_date.desc())
            .first()
        )

        if prediction and selected_cult.planting_date:
            prediction.price_date_95 = selected_cult.planting_date + timedelta(days=95)
            prediction.price_date_105 = selected_cult.planting_date + timedelta(days=105)
            prediction.price_date_115 = selected_cult.planting_date + timedelta(days=115)

        price_sql = text("""
            SELECT price_date, price_per_kg
            FROM kamis_tomato_price
            ORDER BY price_date DESC
            LIMIT 1
        """)
        latest_price_row = db.session.execute(price_sql).fetchone()

        if latest_price_row and prediction:
            prediction.price_date = latest_price_row[0]
            prediction.latest_market_price = latest_price_row[1]

        if farm:
            weather = get_weather(farm.region_l1, farm.region_l2)

        latest_env = (
            EnvSummary.query
            .filter(EnvSummary.cult_id == selected_cult.cult_id)
            .order_by(EnvSummary.measure_date.desc())
            .first()
        )

        current_alert = None
        if farm:
            current_alert = get_weather_alert_status(farm.region_l1, farm.region_l2)

        logs = build_monitoring_logs(
            latest_env,
            weather_alert=current_alert,
            last_measured_at=None
        )

        priority_log = next((log for log in logs if log.get("level") in ["danger", "warning"]), None)

        if not priority_log and logs:
            priority_log = logs[0]

        subquery = (
            db.session.query(
                Growth.plant_num,
                func.max(Growth.inspect_date).label('max_date')
            )
            .filter(Growth.cult_id == selected_cult.cult_id)
            .group_by(Growth.plant_num)
            .subquery()
        )

        latest_growth_list = (
            Growth.query
            .join(
                subquery,
                (Growth.plant_num == subquery.c.plant_num) &
                (Growth.inspect_date == subquery.c.max_date)
            )
            .filter(Growth.cult_id == selected_cult.cult_id)
            .order_by(Growth.plant_num.asc())
            .all()
        )

    return render_template(
        "dashboard.html",
        cult_list=cult_list,
        selected_cult=selected_cult,
        farm=farm,
        weather=weather,
        username=g.user.username,
        prediction=prediction,
        latest_growth_list=latest_growth_list,
        latest_env=latest_env,
        logs=logs,
        priority_log=priority_log,
        today=date.today()
    )