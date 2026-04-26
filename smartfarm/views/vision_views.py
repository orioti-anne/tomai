from flask import Blueprint, render_template, g, redirect, url_for, request, jsonify
from smartfarm.models import Cultivations, Farms
from smartfarm import db
import cv2
import numpy as np
import tempfile
import os
import threading


bp = Blueprint('vision', __name__, url_prefix='/vision')

# 모델 지연 로딩
_disease_model = None
_quality_model = None
_seg_model = None


def _generate_vision_video(session_id, video_bytes, shot_type, output_path):
    """백그라운드 영상 생성"""
    import tempfile, os
    import cv2

    try:
        disease_model, quality_model, seg_model = get_models()

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        SEG_COLOR = {
            'tom_fruit_breaker_poly': (0, 165, 255),
            'tom_fruit_pink_poly':    (147, 20, 255),
            'tom_fruit_red_poly':     (0, 0, 255),
        }
        SEG_LABEL = {
            'tom_fruit_breaker_poly': 'Breaker',
            'tom_fruit_pink_poly':    'Pink',
            'tom_fruit_red_poly':     'Red',
        }
        QUALITY_COLOR = {
            'unripe':    (0, 200, 0),
            'half_ripe': (0, 200, 255),
            'ripe':      (0, 80, 255),
            'rotten':    (120, 120, 120),
        }

        def draw_text_bg(img, text, pos, scale, color):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            x, y = pos
            if y - th - 4 < 0: y = y + th + 4
            if x + tw + 4 > img.shape[1]: x = img.shape[1] - tw - 6
            cv2.rectangle(img, (x, y-th-4), (x+tw+4, y+4), (0,0,0), -1)
            cv2.putText(img, text, (x+2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            overlay = frame.copy()

            # 품질 박스
            if shot_type in ('wide', 'zoom'):
                q = quality_model(frame, conf=0.5, verbose=False)[0]
                for box in q.boxes:
                    cls = q.names[int(box.cls)]
                    x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                    color = QUALITY_COLOR.get(cls, (200,200,200))
                    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 1)
                    draw_text_bg(overlay, f"Q:{cls[:4]} {float(box.conf):.2f}", (x1, y2+15), 0.4, color)

            # 질병 + 세그 박스
            if shot_type == 'zoom':
                d = disease_model(frame, conf=0.5, verbose=False)[0]
                for box in d.boxes:
                    cls = d.names[int(box.cls)]
                    if cls == 'Healthy': continue
                    x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,255), 2)
                    draw_text_bg(overlay, f"D:{cls[:8]}", (x1, y1-8), 0.45, (0,0,255))

                s = seg_model(frame, conf=0.2, verbose=False)[0]
                if s.masks is not None:
                    seg_areas = {k: [] for k in SEG_COLOR}
                    for box, mask in zip(s.boxes, s.masks):
                        cls = s.names[int(box.cls)]
                        if cls in seg_areas:
                            seg_areas[cls].append(int(mask.data.sum().item()))

                    red_areas = seg_areas.get('tom_fruit_red_poly', [])
                    red_avg = sum(red_areas)/len(red_areas) if red_areas else None

                    for box, mask in zip(s.boxes, s.masks):
                        cls = s.names[int(box.cls)]
                        color = SEG_COLOR.get(cls, (255,255,255))
                        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                        label = SEG_LABEL.get(cls, cls)
                        area = int(mask.data.sum().item())
                        cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 3)
                        if red_avg:
                            pct = min(area/red_avg*100, 100)
                            draw_text_bg(overlay, f"S:{label} {pct:.0f}%", (x1, y1-28), 0.5, color)
                            bar_w = x2-x1
                            cv2.rectangle(overlay, (x1,y2+2), (x2,y2+10), (50,50,50), -1)
                            cv2.rectangle(overlay, (x1,y2+2), (x1+int(bar_w*pct/100),y2+10), color, -1)

            result = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
            out.write(result)

        cap.release()
        out.release()
        os.unlink(tmp_path)

        # DB 업데이트
        from sqlalchemy import text
        with db.engine.begin() as conn:
            conn.execute(text("""
                UPDATE vision_session 
                SET video_status='ready', video_path=:path
                WHERE session_id=:sid
            """), {'path': output_path, 'sid': session_id})

    except Exception as e:
        import traceback
        traceback.print_exc()
        from sqlalchemy import text
        with db.engine.begin() as conn:
            conn.execute(text("""
                UPDATE vision_session SET video_status='error' WHERE session_id=:sid
            """), {'sid': session_id})


@bp.route('/video-status/<int:session_id>')
def video_status(session_id):
    try:
        from sqlalchemy import text
        with db.engine.connect() as conn:
            row = conn.execute(text("""
                SELECT video_status, video_path FROM vision_session WHERE session_id=:sid
            """), {'sid': session_id}).fetchone()

        if not row:
            return jsonify({'status': 'not_found'}), 404

        status = row.video_status
        video_path = row.video_path

        if status == 'ready' and video_path:
            # static 경로로 변환
            video_url = '/static/vision_output/' + os.path.basename(video_path)
            return jsonify({'status': 'ready', 'video_url': video_url})
        elif status == 'deleted':
            return jsonify({'status': 'deleted'})
        else:
            return jsonify({'status': 'pending'})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/zones/<int:cult_id>')
def zones(cult_id):
    try:
        from sqlalchemy import text
        with db.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT zone_name FROM vision_session
                WHERE cult_id=:cult_id AND zone_name IS NOT NULL AND zone_name != ''
                ORDER BY zone_name
            """), {'cult_id': cult_id})
            zone_list = [row[0] for row in result]
        return jsonify({'zones': zone_list}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  'static', 'vision_output')
        output_path = os.path.join(output_dir, f'vision_{session_id}.mp4')
        t = threading.Thread(target=_generate_vision_video, args=(session_id, image_or_video, shot_type, output_path))
        t.daemon = True
        t.start()

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
                    'video_path': row.video_path or '',
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