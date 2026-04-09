from datetime import date, datetime, timedelta
import os
import itertools
import joblib
import numpy as np
import pandas as pd

from flask import Blueprint, render_template, request, g, redirect, url_for, flash
from smartfarm import db
from smartfarm.models import Cultivations, Farms, EnvCleaned, Growth
from functools import lru_cache

bp = Blueprint("growth", __name__, url_prefix="/growth")


def get_user_cultivations(user_id):
    return (
        Cultivations.query
        .join(Farms, Cultivations.farm_id == Farms.farm_id)
        .filter(
            Farms.user_id == user_id,
            Farms.is_active == "Y"
        )
        .filter(Cultivations.status != "hidden")
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


def get_latest_environment(cult_id):
    return (
        EnvCleaned.query
        .filter(EnvCleaned.cult_id == cult_id)
        .order_by(EnvCleaned.measure_time.desc())
        .first()
    )


def get_latest_growth_list(cult_id):
    rows = (
        Growth.query
        .filter(Growth.cult_id == cult_id)
        .order_by(
            Growth.inspect_date.desc(),
            Growth.created_at.desc(),
            Growth.growth_id.desc()
        )
        .all()
    )

    latest_by_plant = {}
    for row in rows:
        plant_num = row.plant_num if row.plant_num is not None else 1
        if plant_num not in latest_by_plant:
            latest_by_plant[plant_num] = row

    return [latest_by_plant[key] for key in sorted(latest_by_plant.keys())]


def get_recent_env_7d_avg(cult_id):
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=7)

    rows = (
        EnvCleaned.query
        .filter(
            EnvCleaned.cult_id == cult_id,
            EnvCleaned.measure_time >= start_dt,
            EnvCleaned.measure_time <= end_dt
        )
        .all()
    )

    if not rows:
        return None

    def avg(values):
        vals = [v for v in values if v is not None]
        return float(sum(vals) / len(vals)) if vals else None

    return {
        "in_temp": avg([r.in_temp for r in rows]),
        "in_humidity": avg([r.in_humidity for r in rows]),
        "in_co2": avg([r.in_co2 for r in rows]),
        "out_acc_solar_rad": avg([r.out_acc_solar_rad for r in rows]),
    }


def to_int_or_none(value):
    if value in (None, ""):
        return None
    return int(value)


def to_float_or_none(value):
    if value in (None, ""):
        return None
    return float(value)


def calculate_vpd(temp, humid):
    es = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
    ea = es * (humid / 100.0)
    return es - ea


def humidity_from_temp_vpd(temp, target_vpd):
    es = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
    humid = (1 - (target_vpd / es)) * 100
    humid = max(0, min(100, humid))
    return round(humid, 1)


def get_dap(selected_cultivation):
    if selected_cultivation and selected_cultivation.planting_date:
        return max((date.today() - selected_cultivation.planting_date).days, 1)
    return 1


def get_stress_temp_by_dap(dap):
    if dap <= 30:
        return 32.0
    elif dap <= 70:
        return 30.0
    elif dap <= 110:
        return 28.0
    else:
        return 27.0


@lru_cache(maxsize=1)
def load_growth_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "ml",
        "models",
        "env_growth_model.joblib",
    )
    model_path = os.path.abspath(model_path)

    print(f"[growth] model_path={model_path}")

    if not os.path.exists(model_path):
        print("[growth] env_growth_model.joblib 없음")
        return None, None

    model_data = joblib.load(model_path)
    model = model_data.get("model")
    features = model_data.get("features", [])

    print(f"[growth] model loaded, features={features}")
    return model, features


def recommend_environment(selected_cultivation, latest_growth, latest_env):
    result = {
        "temp": None,
        "humidity": None,
        "co2": None,
        "stress_temp": None,
        "predicted_rgr": None,
    }

    if not latest_growth:
        print("[growth] latest_growth 없음")
        return result

    if latest_growth.plant_height is None:
        print("[growth] plant_height 없음")
        return result

    model, features = load_growth_model()
    if model is None or not features:
        print("[growth] 모델 또는 features 없음")
        return result

    dap = get_dap(selected_cultivation)
    stress_temp = get_stress_temp_by_dap(dap)

    prev_height = float(latest_growth.plant_height or 0)
    leaf_count = float(latest_growth.leaf_count or 0)

    current_solar = 3000.0
    if latest_env and latest_env.out_acc_solar_rad is not None:
        current_solar = float(latest_env.out_acc_solar_rad)

    temp_candidates = [20, 22, 24, 26, 28]
    vpd_candidates = [0.8, 1.0, 1.2, 1.4]
    co2_candidates = [400, 600, 800, 1000, 1200]

    candidates = []

    for temp, vpd, co2 in itertools.product(temp_candidates, vpd_candidates, co2_candidates):
        humidity = humidity_from_temp_vpd(temp, vpd)

        row = {
            "PERIOD_GDD": max(temp - 10, 0),
            "PERIOD_VPD": vpd,
            "PERIOD_SOLAR_ACC": current_solar,
            "VPD_SOLAR_INTERACT": vpd * current_solar,
            "HIGH_TEMP_SUM": 1 if temp >= stress_temp else 0,
            "PERIOD_CO2_AVG": float(co2),
            "PREV_HEIGHT": prev_height,
            "LEAF_COUNT": leaf_count,
            "DAP": float(dap),
        }

        missing_features = [f for f in features if f not in row]
        if missing_features:
            print(f"[growth] missing_features={missing_features}")
            return result

        X = pd.DataFrame([[row[f] for f in features]], columns=features)
        pred = float(model.predict(X)[0])

        candidates.append({
            "temp": round(float(temp), 1),
            "humidity": round(float(humidity), 1),
            "co2": round(float(co2), 0),
            "stress_temp": stress_temp,
            "predicted_rgr": pred,
        })

    if not candidates:
        print("[growth] candidates 없음")
        return result

    best = max(candidates, key=lambda x: x["predicted_rgr"])
    print(f"[growth] best={best}")
    return best


def predict_growth_rgr(selected_cultivation, growth, env_avg):
    if not growth or growth.plant_height is None:
        return None

    model, features = load_growth_model()
    if model is None or not features:
        return None

    dap = get_dap(selected_cultivation)

    temp = float(env_avg.get("in_temp")) if env_avg and env_avg.get("in_temp") is not None else None
    humidity = float(env_avg.get("in_humidity")) if env_avg and env_avg.get("in_humidity") is not None else None
    co2 = float(env_avg.get("in_co2")) if env_avg and env_avg.get("in_co2") is not None else None
    solar = float(env_avg.get("out_acc_solar_rad")) if env_avg and env_avg.get("out_acc_solar_rad") is not None else None

    if temp is None or humidity is None or co2 is None or solar is None:
        print("[growth] env_avg 부족으로 초장 예측 불가")
        return None

    vpd = calculate_vpd(temp, humidity)
    stress_temp = get_stress_temp_by_dap(dap)

    row = {
        "PERIOD_GDD": max(temp - 10, 0),
        "PERIOD_VPD": vpd,
        "PERIOD_SOLAR_ACC": solar,
        "VPD_SOLAR_INTERACT": vpd * solar,
        "HIGH_TEMP_SUM": 1 if temp >= stress_temp else 0,
        "PERIOD_CO2_AVG": co2,
        "PREV_HEIGHT": float(growth.plant_height or 0),
        "LEAF_COUNT": float(growth.leaf_count or 0),
        "DAP": float(dap),
    }

    missing_features = [f for f in features if f not in row]
    if missing_features:
        print(f"[growth] missing_features in forecast={missing_features}")
        return None

    X = pd.DataFrame([[row[f] for f in features]], columns=features)
    pred = float(model.predict(X)[0])
    print(f"[growth] forecast_rgr plant_num={growth.plant_num}, pred={pred}")
    return pred


def build_height_forecast(selected_cultivation, growth, env_avg):
    if not growth or growth.plant_height is None:
        return None

    predicted_rgr = predict_growth_rgr(selected_cultivation, growth, env_avg)
    if predicted_rgr is None:
        return None

    dap = get_dap(selected_cultivation)
    predicted_rgr = min(predicted_rgr, 0.01)
    if dap < 40:
        decay_factor = 1.0
    elif dap < 70:
        decay_factor = 0.85
    else:
        decay_factor = 0.7

    current_height = float(growth.plant_height)
    predicted_7d = current_height * ((1 + predicted_rgr) ** 7)
    predicted_28d = current_height + ((current_height * predicted_rgr * 28) * decay_factor)

    return {
        "predicted_7d": round(predicted_7d, 1),
        "predicted_28d": round(predicted_28d, 1),
    }


@bp.route("/", methods=["GET", "POST"])
def growth_monitoring():
    if not g.user:
        return redirect(url_for("auth.login"))

    cultivations = get_user_cultivations(g.user.user_id)
    selected_cult_id = request.values.get("cult_id", type=int)
    selected_cultivation = get_selected_cultivation(cultivations, selected_cult_id)

    if request.method == "POST":
        if not selected_cultivation:
            flash("선택된 재배 정보가 없습니다.", "error")
            return redirect(url_for("growth.growth_monitoring"))

        inspect_date_str = (request.form.get("inspect_date") or "").strip()
        plant_num_str = (request.form.get("plant_num") or "").strip()
        plant_height_str = (request.form.get("plant_height") or "").strip()

        if not inspect_date_str:
            flash("점검일은 필수입니다.", "error")
            return redirect(url_for("growth.growth_monitoring", cult_id=selected_cultivation.cult_id))

        if not plant_num_str:
            flash("개체번호는 필수입니다.", "error")
            return redirect(url_for("growth.growth_monitoring", cult_id=selected_cultivation.cult_id))

        if not plant_height_str:
            flash("초장은 필수입니다.", "error")
            return redirect(url_for("growth.growth_monitoring", cult_id=selected_cultivation.cult_id))

        try:
            inspect_date = datetime.strptime(inspect_date_str, "%Y-%m-%d").date()
            plant_num = to_int_or_none(plant_num_str)

            growth = (
                Growth.query
                .filter(
                    Growth.cult_id == selected_cultivation.cult_id,
                    Growth.inspect_date == inspect_date,
                    Growth.plant_num == plant_num
                )
                .first()
            )

            is_new = growth is None

            if is_new:
                growth = Growth(
                    cult_id=selected_cultivation.cult_id,
                    inspect_date=inspect_date,
                    plant_num=plant_num,
                    blooming_group=None,
                    fruiting_group=None,
                    created_at=date.today(),
                )
                db.session.add(growth)

            growth.branch_num = to_int_or_none(request.form.get("branch_num"))
            growth.plant_height = to_float_or_none(request.form.get("plant_height"))
            growth.growth_length = to_float_or_none(request.form.get("growth_length"))
            growth.leaf_count = to_int_or_none(request.form.get("leaf_count"))
            growth.leaf_length = to_float_or_none(request.form.get("leaf_length"))
            growth.leaf_width = to_float_or_none(request.form.get("leaf_width"))
            growth.branch_width = to_float_or_none(request.form.get("branch_width"))
            growth.cluster_height = to_float_or_none(request.form.get("cluster_height"))
            growth.cluster_num = to_int_or_none(request.form.get("cluster_num"))
            growth.flowers_per_cluster = to_int_or_none(request.form.get("flowers_per_cluster"))
            growth.blooming_per_cluster = to_int_or_none(request.form.get("blooming_per_cluster"))
            growth.fruits_per_cluster = to_int_or_none(request.form.get("fruits_per_cluster"))
            growth.remarks = (request.form.get("remarks") or "").strip() or None

            growth.blooming_group = None
            growth.fruiting_group = None

            db.session.commit()

            if is_new:
                flash("생육 상태가 등록되었습니다.", "success")
            else:
                flash("동일 개체번호의 생육 상태가 수정되었습니다.", "success")

            return redirect(url_for("growth.growth_monitoring", cult_id=selected_cultivation.cult_id))

        except Exception as e:
            db.session.rollback()
            flash(f"생육 상태 저장 중 오류가 발생했습니다: {e}", "error")
            return redirect(url_for("growth.growth_monitoring", cult_id=selected_cultivation.cult_id))

    latest_env = None
    latest_growth_list = []
    latest_growth = None

    recommended_env = {
        "temp": None,
        "humidity": None,
        "co2": None,
        "stress_temp": None,
        "predicted_rgr": None,
    }

    growth_forecasts = {}

    if selected_cultivation:
        latest_env = get_latest_environment(selected_cultivation.cult_id)
        latest_growth_list = get_latest_growth_list(selected_cultivation.cult_id)
        latest_growth = latest_growth_list[0] if latest_growth_list else None
        env_avg_7d = get_recent_env_7d_avg(selected_cultivation.cult_id)

        recommended_env = recommend_environment(selected_cultivation, latest_growth, latest_env)

        for growth in latest_growth_list:
            key = growth.plant_num if growth.plant_num is not None else 1
            growth_forecasts[key] = build_height_forecast(selected_cultivation, growth, env_avg_7d)

    return render_template(
        "growth.html",
        cultivations=cultivations,
        selected_cultivation=selected_cultivation,
        latest_env=latest_env,
        latest_growth=latest_growth,
        latest_growth_list=latest_growth_list,
        today=date.today(),
        recommended_env=recommended_env,
        growth_forecasts=growth_forecasts,
    )




@bp.route("/get_latest_plant_data/<int:cult_id>/<int:plant_num>")
def get_latest_plant_data(cult_id, plant_num):
    latest = (
        Growth.query
        .filter(Growth.cult_id == cult_id, Growth.plant_num == plant_num)
        .order_by(Growth.inspect_date.desc(), Growth.created_at.desc())
        .first()
    )

    if latest:
        return {
            "success": True,
            "data": {
                "branch_num": latest.branch_num,
                "plant_height": latest.plant_height,
                "growth_length": latest.growth_length,
                "leaf_count": latest.leaf_count,
                "leaf_length": latest.leaf_length,
                "leaf_width": latest.leaf_width,
                "branch_width": latest.branch_width,
                "cluster_height": latest.cluster_height,
                "cluster_num": latest.cluster_num,
                "flowers_per_cluster": latest.flowers_per_cluster,
                "blooming_per_cluster": latest.blooming_per_cluster,
                "fruits_per_cluster": latest.fruits_per_cluster,
                "remarks": latest.remarks
            }
        }
    return {"success": False}