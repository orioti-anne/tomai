import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()



from smartfarm import create_app, db
from smartfarm.models import Cultivations, Farms, EnvSummary, EnvCleaned

app = create_app(enable_scheduler=True)

from flask import jsonify, request
from datetime import datetime
from sqlalchemy import func

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

def get_latest_daily_summary(cult_id):
    return EnvSummary.query.filter(EnvSummary.cult_id == cult_id).order_by(EnvSummary.measure_date.desc()).first()

def get_previous_daily_summary(cult_id, latest_date):
    if not latest_date:
        return None
    return EnvSummary.query.filter(EnvSummary.cult_id == cult_id, EnvSummary.measure_date < latest_date).order_by(EnvSummary.measure_date.desc()).first()

def get_daily_chart_rows(cult_id, days=7):
    rows = EnvSummary.query.filter(EnvSummary.cult_id == cult_id).order_by(EnvSummary.measure_date.desc()).limit(days).all()
    return list(reversed(rows))

def get_latest_hourly_base_date(cult_id):
    return EnvCleaned.query.with_entities(func.max(EnvCleaned.measure_date)).filter(EnvCleaned.cult_id == cult_id).scalar()

def get_hourly_chart_rows(cult_id, base_date):
    if not base_date:
        return []
    return EnvCleaned.query.filter(EnvCleaned.cult_id == cult_id, EnvCleaned.measure_date == base_date).order_by(EnvCleaned.measure_time.asc()).all()

def build_change_info(curr, prev, unit="", percent=False):
    if curr is None or prev is None:
        return None
    curr, prev = float(curr), float(prev)
    diff = curr - prev
    if percent:
        if prev == 0:
            return None
        rate = (diff / prev) * 100
        return {"direction": "up" if rate > 0 else "down" if rate < 0 else "same",
                "value": abs(rate), "text": f"{abs(rate):.1f}% 어제 대비"}
    return {"direction": "up" if diff > 0 else "down" if diff < 0 else "same",
            "value": abs(diff), "text": f"{abs(diff):.1f}{unit} 어제 대비"}

def build_monitoring_logs(latest_env, weather_alert=None, last_measured_at=None):
    logs = []
    if weather_alert:
        logs.append({"level": "danger", "title": f"🚨 {weather_alert.get('title', '기상 특보 발령')}",
                     "message": weather_alert.get('message', ''), "time_text": "실시간"})
    if not latest_env:
        return logs
    date_text = latest_env.measure_date.strftime("%m/%d") if latest_env.measure_date else "-"
    if latest_env.daily_in_temp is not None:
        if latest_env.daily_in_temp >= 28:
            logs.append({"level": "danger", "title": "🌡️ 고온 주의",
                         "message": f"평균 온도가 {latest_env.daily_in_temp:.1f}°C로 높습니다.", "time_text": date_text})
        elif latest_env.daily_in_temp <= 12:
            logs.append({"level": "warning", "title": "❄️ 저온 주의",
                         "message": f"야간 온도 하락에 대비하세요. (현재 {latest_env.daily_in_temp:.1f}°C)", "time_text": date_text})
        else:
            logs.append({"level": "success", "title": "온도 최적 상태",
                         "message": f"실내 온도({latest_env.daily_in_temp:.1f}°C)가 생육에 적합한 범위 내에 있습니다.", "time_text": date_text})
    if latest_env.daily_in_humidity is not None:
        if 55 <= latest_env.daily_in_humidity <= 75:
            logs.append({"level": "success", "title": "습도 적정",
                         "message": f"실내 습도({latest_env.daily_in_humidity:.1f}%)가 안정적입니다.", "time_text": date_text})
        else:
            logs.append({"level": "warning", "title": "💧 습도 관리 필요",
                         "message": "습도가 적정 범위를 벗어났습니다.", "time_text": date_text})
    full_time_text = last_measured_at.strftime("%m/%d %H:%M") if last_measured_at else date_text
    logs.append({"level": "info", "title": "시스템 알림",
                 "message": "환경센서가 최신 데이터를 성공적으로 수집하였습니다.", "time_text": full_time_text})
    return logs[:10]

@app.route("/api/monitoring/<int:cult_id>")
def api_monitoring(cult_id):
    try:
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
            "hourly_base_date": hourly_base_date.strftime("%Y-%m-%d") if hourly_base_date else None,
            "logs": logs,
            "temp_change": build_change_info(
                latest_env.daily_in_temp if latest_env else None,
                previous_env.daily_in_temp if previous_env else None, unit="°C"),
            "humidity_change": build_change_info(
                latest_env.daily_in_humidity if latest_env else None,
                previous_env.daily_in_humidity if previous_env else None, unit="%"),
            "co2_change": build_change_info(
                latest_env.daily_in_co2 if latest_env else None,
                previous_env.daily_in_co2 if previous_env else None, unit=" ppm"),
            "solar_change": build_change_info(
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


from ultralytics import YOLO
import cv2
import numpy as np

_DL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smartfarm', 'dl', 'models')
_disease_model = YOLO(os.path.join(_DL_DIR, 'disease_best.pt'))
_quality_model  = YOLO(os.path.join(_DL_DIR, 'quality_best.pt'))
_seg_model      = YOLO(os.path.join(_DL_DIR, 'seg_best.pt'))


def _run_vision(image_or_video, shot_type):
    """영상 전체 프레임 분석 후 집계 반환"""
    import tempfile, os

    # 파일로 저장 후 VideoCapture로 열기
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(image_or_video)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 집계용
    quality_total = {}
    disease_total = {}
    disease_conf = {}
    seg_total = {}

    frame_count = 0
    analyzed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 15 != 0:  # 15프레임마다 분석
            continue
        analyzed += 1

        if shot_type in ('wide', 'zoom'):
            # 품질 모델
            q = _quality_model(frame, conf=0.5, verbose=False)[0]
            for box in q.boxes:
                cls = q.names[int(box.cls)]
                quality_total[cls] = quality_total.get(cls, 0) + 1

        if shot_type == 'zoom':
            # 질병 모델
            d = _disease_model(frame, conf=0.5, verbose=False)[0]
            for box in d.boxes:
                cls = d.names[int(box.cls)]
                if cls == 'Healthy':
                    continue
                disease_total[cls] = disease_total.get(cls, 0) + 1
                disease_conf.setdefault(cls, []).append(float(box.conf))

            # 세그 모델
            s = _seg_model(frame, conf=0.4, verbose=False)[0]
            if s.masks is not None:
                for box, mask in zip(s.boxes, s.masks):
                    cls = s.names[int(box.cls)]
                    area = int(mask.data.sum().item())
                    seg_total.setdefault(cls, []).append(area)

    cap.release()
    os.unlink(tmp_path)

    # 집계 결과 정리
    results = {}
    q_total = sum(quality_total.values()) or 1
    results['quality'] = [
        {'class_name': k, 'count': v, 'ratio': round(v / q_total * 100, 2)}
        for k, v in quality_total.items()
    ]

    results['disease'] = [
        {'class_name': k, 'count': v,
         'avg_conf': round(sum(disease_conf[k]) / len(disease_conf[k]), 3)}
        for k, v in disease_total.items()
    ]

    red_areas = seg_total.get('tom_fruit_red_poly', [])
    red_avg = sum(red_areas) / len(red_areas) if red_areas else None
    results['segment'] = []
    for cls, areas in seg_total.items():
        avg_area = sum(areas) / len(areas)
        avg_growth = round(avg_area / red_avg * 100, 2) if red_avg else None
        results['segment'].append({
            'class_name': cls,
            'count': len(areas),
            'avg_area': round(avg_area, 2),
            'avg_growth': avg_growth
        })

    results['analyzed_frames'] = analyzed
    results['total_frames'] = total_frames
    return results

@app.route("/api/vision/analyze/<int:cult_id>", methods=["POST"])
def api_vision_analyze(cult_id):
    try:
        shot_type = request.form.get('shot_type', 'wide')  # wide or zoom
        if 'image' not in request.files:
            return jsonify({"error": "이미지가 없습니다"}), 400

        file = request.files['image']
        image_or_video = file.read()  # bytes로 읽기
        vision_results = _run_vision(image_or_video, shot_type)

        # DB 저장
        from sqlalchemy import text
        with db.engine.begin() as conn:
            # 세션 생성
            row = conn.execute(text("""
                INSERT INTO vision_session (cult_id, shot_type, total_frames)
                VALUES (:cult_id, :shot_type, 1)
                RETURNING session_id
            """), {'cult_id': cult_id, 'shot_type': shot_type})
            session_id = row.fetchone()[0]

            # 품질 저장
            for q in vision_results.get('quality', []):
                conn.execute(text("""
                    INSERT INTO vision_quality (session_id, class_name, count, ratio)
                    VALUES (:sid, :cls, :cnt, :ratio)
                """), {'sid': session_id, 'cls': q['class_name'], 'cnt': q['count'], 'ratio': q['ratio']})

            # 질병 저장
            for d in vision_results.get('disease', []):
                conn.execute(text("""
                    INSERT INTO vision_disease (session_id, class_name, count, avg_conf)
                    VALUES (:sid, :cls, :cnt, :conf)
                """), {'sid': session_id, 'cls': d['class_name'], 'cnt': d['count'], 'conf': d['avg_conf']})

            # 세그 저장
            for s in vision_results.get('segment', []):
                conn.execute(text("""
                    INSERT INTO vision_segment (session_id, class_name, count, avg_area, avg_growth)
                    VALUES (:sid, :cls, :cnt, :area, :growth)
                """), {'sid': session_id, 'cls': s['class_name'], 'cnt': s['count'],
                       'area': s['avg_area'], 'growth': s['avg_growth']})

        return jsonify({
            "session_id": session_id,
            "shot_type": shot_type,
            "results": vision_results
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/vision/history/<int:cult_id>")
def api_vision_history(cult_id):
    """Vision 분석 이력 조회"""
    try:
        from sqlalchemy import text
        with db.engine.connect() as conn:
            sessions = conn.execute(text("""
                SELECT s.session_id, s.shot_type, s.analyzed_at,
                       json_agg(DISTINCT jsonb_build_object(
                           'class_name', q.class_name, 'count', q.count, 'ratio', q.ratio
                       )) FILTER (WHERE q.id IS NOT NULL) as quality,
                       json_agg(DISTINCT jsonb_build_object(
                           'class_name', d.class_name, 'count', d.count, 'avg_conf', d.avg_conf
                       )) FILTER (WHERE d.id IS NOT NULL) as disease,
                       json_agg(DISTINCT jsonb_build_object(
                           'class_name', sg.class_name, 'count', sg.count,
                           'avg_growth', sg.avg_growth
                       )) FILTER (WHERE sg.id IS NOT NULL) as segment
                FROM vision_session s
                LEFT JOIN vision_quality q ON s.session_id = q.session_id
                LEFT JOIN vision_disease d ON s.session_id = d.session_id
                LEFT JOIN vision_segment sg ON s.session_id = sg.session_id
                WHERE s.cult_id = :cult_id
                GROUP BY s.session_id
                ORDER BY s.analyzed_at DESC
                LIMIT 20
            """), {'cult_id': cult_id})

            history = []
            for row in sessions:
                history.append({
                    'session_id': row.session_id,
                    'shot_type': row.shot_type,
                    'analyzed_at': row.analyzed_at.strftime("%Y-%m-%d %H:%M"),
                    'quality': row.quality or [],
                    'disease': row.disease or [],
                    'segment': row.segment or []
                })

        return jsonify({'cult_id': cult_id, 'history': history}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500





@app.route("/api/sync/run")
def run_sync():
    try:
        from smartfarm.services.cloud_sync_service import run_full_sync
        run_full_sync(app)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/prediction/run", methods=["POST"])
def api_prediction_run():
    key = request.headers.get("X-API-Key")
    if key != API_SECRET:
        return jsonify({"error": "unauthorized"}), 401
    try:
        data = request.get_json()
        cult_id = data.get("cult_id")
        user_id = data.get("user_id")
        farm_id = data.get("farm_id")
        from smartfarm.services.prediction_service import run_default_prediction, run_ml_prediction
        result = run_ml_prediction(cult_id)
        if not result:
            result = run_default_prediction(
                cult_id=cult_id,
                farm_id=farm_id,
                planting_date=data.get("planting_date"),
                item=data.get("item"),
                crop_cycle=data.get("crop_cycle"),
                item_variety=data.get("item_variety"),
                planting_area=data.get("planting_area"),
                planting_density=data.get("planting_density"),
                house_type=data.get("house_type"),
                house_form=data.get("house_form")
            )
        return jsonify({"status": "success", "result": result}), 200
    except Exception as e:
        print(f"[API PREDICTION ERROR] {e}")
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)