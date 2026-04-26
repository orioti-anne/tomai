from flask import Blueprint, render_template, g, redirect, url_for, request, jsonify
from smartfarm.models import Cultivations, Farms
from smartfarm import db
import cv2
import numpy as np
import tempfile
import os

bp = Blueprint('vision', __name__, url_prefix='/vision')

# 모델 지연 로딩
_disease_model = None
_quality_model = None
_seg_model = None

def get_models():
    global _disease_model, _quality_model, _seg_model
    if _disease_model is None:
        from ultralytics import YOLO
        _DL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dl', 'models')
        _disease_model = YOLO(os.path.join(_DL_DIR, 'disease_best.pt'))
        _quality_model = YOLO(os.path.join(_DL_DIR, 'quality_best.pt'))
        _seg_model = YOLO(os.path.join(_DL_DIR, 'seg_best.pt'))
    return _disease_model, _quality_model, _seg_model


def _run_vision(image_or_video, shot_type):
    disease_model, quality_model, seg_model = get_models()

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(image_or_video)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
        if frame_count % 15 != 0:
            continue
        analyzed += 1

        if shot_type in ('wide', 'zoom'):
            q = quality_model(frame, conf=0.5, verbose=False)[0]
            for box in q.boxes:
                cls = q.names[int(box.cls)]
                quality_total[cls] = quality_total.get(cls, 0) + 1

        if shot_type == 'zoom':
            d = disease_model(frame, conf=0.5, verbose=False)[0]
            for box in d.boxes:
                cls = d.names[int(box.cls)]
                if cls == 'Healthy':
                    continue
                disease_total[cls] = disease_total.get(cls, 0) + 1
                disease_conf.setdefault(cls, []).append(float(box.conf))

            s = seg_model(frame, conf=0.4, verbose=False)[0]
            if s.masks is not None:
                for box, mask in zip(s.boxes, s.masks):
                    cls = s.names[int(box.cls)]
                    area = int(mask.data.sum().item())
                    seg_total.setdefault(cls, []).append(area)

    cap.release()
    os.unlink(tmp_path)

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


@bp.route('/')
def index():
    if not g.user:
        return redirect(url_for('auth.login'))

    cult_list = (
        Cultivations.query
        .join(Farms, Cultivations.farm_id == Farms.farm_id)
        .filter(
            Farms.user_id == g.user.user_id,
            Cultivations.status == 'active'
        )
        .all()
    )
    selected_cult = cult_list[0] if cult_list else None

    return render_template('vision.html',
        cult_list=cult_list,
        selected_cult=selected_cult,
        username=g.user.username
    )

@bp.route('/zones/<int:cult_id>')
def zones(cult_id):
    try:
        from sqlalchemy import text
        with db.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT zone_name 
                FROM vision_session 
                WHERE cult_id = :cult_id 
                AND zone_name IS NOT NULL 
                AND zone_name != ''
                ORDER BY zone_name
            """), {'cult_id': cult_id})
            zones = [row[0] for row in result]
        return jsonify({'zones': zones}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@bp.route('/analyze/<int:cult_id>', methods=['POST'])
def analyze(cult_id):
    if not g.user:
        return jsonify({"error": "로그인이 필요합니다"}), 401
    try:
        shot_type = request.form.get('shot_type', 'wide')
        if 'image' not in request.files:
            return jsonify({"error": "이미지가 없습니다"}), 400

        file = request.files['image']
        image_or_video = file.read()
        vision_results = _run_vision(image_or_video, shot_type)

        from sqlalchemy import text
        with db.engine.begin() as conn:
            row = conn.execute(text("""
                INSERT INTO vision_session (cult_id, shot_type, total_frames, image_path, zone_name)
                VALUES (:cult_id, :shot_type, :total_frames, :image_path, :zone_name)
                RETURNING session_id
            """), {
                'cult_id': cult_id,
                'shot_type': shot_type,
                'total_frames': vision_results.get('total_frames', 0),
                'image_path': file.filename,
                'zone_name': request.form.get('zone_name', '')
            })
            session_id = row.fetchone()[0]

            for q in vision_results.get('quality', []):
                conn.execute(text("""
                    INSERT INTO vision_quality (session_id, class_name, count, ratio)
                    VALUES (:sid, :cls, :cnt, :ratio)
                """), {'sid': session_id, 'cls': q['class_name'], 'cnt': q['count'], 'ratio': q['ratio']})

            for d in vision_results.get('disease', []):
                conn.execute(text("""
                    INSERT INTO vision_disease (session_id, class_name, count, avg_conf)
                    VALUES (:sid, :cls, :cnt, :conf)
                """), {'sid': session_id, 'cls': d['class_name'], 'cnt': d['count'], 'conf': d['avg_conf']})

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


@bp.route('/history/<int:cult_id>')
def history(cult_id):
    try:
        from sqlalchemy import text
        with db.engine.connect() as conn:
            sessions = conn.execute(text("""
                SELECT s.session_id, s.shot_type, s.analyzed_at, s.zone_name,
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
                    'zone_name': row.zone_name or '-',
                    'quality': row.quality or [],
                    'disease': row.disease or [],
                    'segment': row.segment or []
                })

        return jsonify({'cult_id': cult_id, 'history': history}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500