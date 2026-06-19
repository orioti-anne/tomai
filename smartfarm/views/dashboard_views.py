import json
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
            # 하위 호환 날짜 (구버전 prediction 대비)
            prediction.price_date_95 = selected_cult.planting_date + timedelta(days=95)
            prediction.price_date_105 = selected_cult.planting_date + timedelta(days=105)
            prediction.price_date_115 = selected_cult.planting_date + timedelta(days=115)

            # comparison_json → 동적 시나리오 데이터 파싱
            raw_cmp = getattr(prediction, 'comparison_json', None)
            if raw_cmp:
                try:
                    cmp_list = json.loads(raw_cmp)
                    prediction.comparison_data = [
                        {
                            **item,
                            "date": (selected_cult.planting_date + timedelta(days=item["days"])).strftime("%m.%d"),
                            "date_full": (selected_cult.planting_date + timedelta(days=item["days"])).strftime("%Y.%m.%d"),
                        }
                        for item in cmp_list
                    ]
                except Exception:
                    prediction.comparison_data = []
            else:
                # 구버전 fallback: 95/105/115 컬럼에서 재구성
                prediction.comparison_data = [
                    {"days": 95,  "price": prediction.price_day_95 or 0,  "date": prediction.price_date_95.strftime("%m.%d"), "date_full": prediction.price_date_95.strftime("%Y.%m.%d")},
                    {"days": 105, "price": prediction.price_day_105 or 0, "date": prediction.price_date_105.strftime("%m.%d"), "date_full": prediction.price_date_105.strftime("%Y.%m.%d")},
                    {"days": 115, "price": prediction.price_day_115 or 0, "date": prediction.price_date_115.strftime("%m.%d"), "date_full": prediction.price_date_115.strftime("%Y.%m.%d")},
                ]

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

    # confidence_score:
    #   floor=55(재배+시세) ~ ceiling=80(R²_max=0.7981, 환경+생육 완비)
    #   구간 25p를 env/grow feature importance 비율로 배분
    #   ENV_W_norm=0.397, GROW_W_norm=0.603  (env=0.295, grow=0.447 기반)
    FLOOR   = 55
    CEILING = 80
    ENV_W_NORM  = 0.397
    GROW_W_NORM = 0.603

    confidence_score = 0
    env_comp = 0.0
    grow_comp = 0.0
    if prediction and selected_cult:
        src = prediction.prediction_source or ''
        if src == 'optimized_model':
            days_elapsed = max(1, (date.today() - selected_cult.planting_date).days) if selected_cult.planting_date else 1

            # 환경 완성도: 실제 기록일 / 경과일
            env_days = db.session.query(func.count(EnvSummary.envsu_id)).filter(
                EnvSummary.cult_id == selected_cult.cult_id
            ).scalar() or 0
            env_comp = min(1.0, env_days / days_elapsed)

            # 생육 완성도: 마지막 조사일 기준 최신성 (v5 모델 피처 중요도 기반)
            # stable(span/n_dates/_mean 등)=71%, _final 측정값=29% → 하한 0.71
            last_inspect = db.session.query(func.max(Growth.inspect_date)).filter(
                Growth.cult_id == selected_cult.cult_id
            ).scalar()
            if last_inspect:
                days_since = (date.today() - last_inspect).days
                recency = max(0.0, 1.0 - days_since / 90)
                grow_comp = round(0.71 + 0.29 * recency, 4)
            else:
                grow_comp = 0.0

            score = FLOOR + (CEILING - FLOOR) * (ENV_W_NORM * env_comp + GROW_W_NORM * grow_comp)
            confidence_score = round(score)
        else:
            confidence_score = 55  # 가격 모델 기반 폴백

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
        today=date.today(),
        confidence_score=confidence_score,
        env_comp=env_comp,
        grow_comp=grow_comp,
    )