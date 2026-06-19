from datetime import datetime, timedelta
from sqlalchemy import text

from smartfarm import db
from smartfarm.ml.predictors.env_growth_predictor import predict_growth_from_env
from smartfarm.ml.predictors.production_predictor import predict_production
from smartfarm.ml.predictors.price_predictor import predict_price


def _row_to_dict(row):
    if row is None:
        return None
    return dict(row._mapping)


def _get_cultivation_base(cult_id: int):
    query = text("""
        SELECT
            c.cult_id,
            c.farm_id,
            c.planting_date,
            c.item,
            c.item_variety,
            f.region_l1,
            f.region_l2
        FROM cultivations c
        LEFT JOIN farms f ON c.farm_id = f.farm_id
        WHERE c.cult_id = :cult_id
    """)
    row = db.session.execute(query, {"cult_id": cult_id}).fetchone()
    return _row_to_dict(row)


def _get_latest_growth(cult_id: int):
    query = text("""
        SELECT
            cult_id,
            inspect_date,
            plant_height,
            leaf_count,
            growth_length,
            leaf_length,
            leaf_width,
            branch_width,
            cluster_height,
            cluster_num,
            flowers_per_cluster,
            blooming_per_cluster,
            fruits_per_cluster,
            growth_days
        FROM grow_summary
        WHERE cult_id = :cult_id
        ORDER BY inspect_date DESC
        FETCH FIRST 1 ROWS ONLY
    """)
    row = db.session.execute(query, {"cult_id": cult_id}).fetchone()
    data = _row_to_dict(row)
    if not data:
        return None

    return {
        "inspect_date": data.get("inspect_date"),
        "plant_height": data.get("plant_height"),
        "leaf_count": data.get("leaf_count"),
        "growth_length": data.get("growth_length"),
        "leaf_length": data.get("leaf_length"),
        "leaf_width": data.get("leaf_width"),
        "branch_width": data.get("branch_width"),
        "cluster_height": data.get("cluster_height"),
        "cluster_num": data.get("cluster_num"),
        "flowers_per_cluster": data.get("flowers_per_cluster"),
        "blooming_per_cluster": data.get("blooming_per_cluster"),
        "fruits_per_cluster": data.get("fruits_per_cluster"),
        "growth_days": data.get("growth_days"),
    }


def _get_recent_env_summary(cult_id: int, days: int = 7):
    query = text("""
        SELECT
            AVG(in_temp) AS avg_temp,
            AVG(in_humidity) AS avg_humid,
            AVG(in_co2) AS avg_co2,
            AVG(out_solar_rad) AS daily_solar,
            SUM(CASE WHEN in_temp >= 30 THEN 1 ELSE 0 END) AS high_temp_hours
        FROM env_cleaned
        WHERE cult_id = :cult_id
          AND measure_date >= CURRENT_DATE - INTERVAL '1 day' * :days
    """)
    row = db.session.execute(query, {"cult_id": cult_id, "days": days}).fetchone()
    return _row_to_dict(row) or {}


def _get_price_history():
    query = text("""
        SELECT *
        FROM (
            SELECT
                price_per_kg,
                price_date,
                LAG(price_per_kg, 1) OVER (ORDER BY price_date) AS prev_per_kg_1d,
                AVG(price_per_kg) OVER (
                    ORDER BY price_date
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS price_ma_3d,
                price_per_kg - LAG(price_per_kg, 1) OVER (ORDER BY price_date) AS price_diff
            FROM kamis_tomato_price
        )
        ORDER BY price_date DESC
        FETCH FIRST 1 ROWS ONLY
    """)
    row = db.session.execute(query).fetchone()
    data = _row_to_dict(row)
    if not data:
        return {}

    return {
        "prev_per_kg_1d": data.get("prev_per_kg_1d") or data.get("price_per_kg"),
        "price_ma_3d": data.get("price_ma_3d") or data.get("price_per_kg"),
        "price_diff": data.get("price_diff") or 0.0,
    }


def _get_weather_summary_for_price():
    """
    현재는 DB weather_index에서 최근값을 가져오는 간단 버전.
    이후 target_date 기준 예측값 또는 API 연동으로 고도화 가능.
    """
    query = text("""
        SELECT *
        FROM weather_index
        ORDER BY w_date DESC
        FETCH FIRST 1 ROWS ONLY
    """)
    row = db.session.execute(query).fetchone()
    data = _row_to_dict(row)
    if not data:
        return {}

    return {
        "avg_temp": data.get("avg_temp"),
        "sunshine": data.get("sunshine"),
        "temp_lag7": data.get("avg_temp"),
    }


def _estimate_dap(planting_date, latest_growth):
    if latest_growth and latest_growth.get("growth_days") is not None:
        return int(latest_growth["growth_days"])

    if planting_date:
        today = datetime.today().date()
        return max((today - planting_date).days, 0)

    return 0


def _merge_growth_with_env_prediction(latest_growth, env_growth_pred):
    """
    실측 생육값이 있으면 우선 사용.
    다만 plant_height가 없으면 env 예측치를 참고해 보정할 수 있도록 구조만 둠.
    """
    result = latest_growth.copy() if latest_growth else {}

    if result.get("plant_height") is None and env_growth_pred:
        pred_rgr = env_growth_pred.get("predicted_rgr_height")
        base_height = 100.0
        result["plant_height"] = base_height * (1 + pred_rgr) if pred_rgr is not None else base_height

    return result


def predict_tomato_cycle(cult_id: int) -> dict:
    cultivation = _get_cultivation_base(cult_id)
    if not cultivation:
        raise ValueError(f"cult_id={cult_id} 에 해당하는 재배정보를 찾을 수 없습니다.")

    latest_growth = _get_latest_growth(cult_id)
    env_summary = _get_recent_env_summary(cult_id, days=7)

    dap = _estimate_dap(
        planting_date=cultivation.get("planting_date"),
        latest_growth=latest_growth,
    )

    env_growth_pred = None
    try:
        env_growth_pred = predict_growth_from_env(
            env_summary=env_summary,
            latest_growth=latest_growth,
            dap=dap,
        )
    except Exception:
        env_growth_pred = None

    merged_growth = _merge_growth_with_env_prediction(
        latest_growth=latest_growth,
        env_growth_pred=env_growth_pred,
    )

    production_pred = predict_production(
        growth_data=merged_growth,
        dap=dap,
    )

    target_date = datetime.today() + timedelta(days=7)

    price_pred = predict_price(
        target_date=target_date,
        market_meta={
            "grade_score": 3,
            "unit_kg": 5,
        },
        price_history=_get_price_history(),
        weather_summary=_get_weather_summary_for_price(),
    )

    expected_quantity = production_pred["predicted_quantity"]
    expected_price_per_kg = price_pred["predicted_price_per_kg"]
    expected_sales = expected_quantity * expected_price_per_kg

    return {
        "cult_id": cult_id,
        "farm_id": cultivation.get("farm_id"),
        "item": cultivation.get("item"),
        "item_variety": cultivation.get("item_variety"),
        "dap": dap,
        "expected_harvest_date": target_date.date().isoformat(),
        "expected_quantity": round(expected_quantity, 2),
        "expected_price_per_kg": round(expected_price_per_kg, 0),
        "expected_sales": round(expected_sales, 0),
        "growth_source": "measured" if latest_growth else "env_predicted",
        "env_growth_prediction": env_growth_pred,
        "production_prediction": production_pred,
        "price_prediction": price_pred,
    }