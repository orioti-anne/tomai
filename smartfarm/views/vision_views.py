from flask import Blueprint, render_template, g, redirect, url_for, request, jsonify, current_app
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
_inspector_model = None

def get_models():
    global _disease_model, _quality_model, _seg_model, _inspector_model
    if _disease_model is None:
        from ultralytics import YOLO
        _DL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dl', 'models')
        _disease_model = YOLO(os.path.join(_DL_DIR, 'disease_best.pt'))
        _quality_model = YOLO(os.path.join(_DL_DIR, 'quality_best.pt'))
        _seg_model = YOLO(os.path.join(_DL_DIR, 'seg_best.pt'))
        _inspector_model = YOLO(os.path.join(_DL_DIR, 'inspector_best.pt'))

        # M4 MPS GPU 사용
        try:
            _disease_model.to('mps')
            _quality_model.to('mps')
            _seg_model.to('mps')
            _inspector_model.to('mps')
            print("[VISION] MPS GPU 사용")
        except Exception as e:
            print(f"[VISION] MPS 사용 불가, CPU 사용: {e}")

    return _disease_model, _quality_model, _seg_model, _inspector_model


def _run_vision(image_or_video, shot_type, is_image=False):
    disease_model, quality_model, seg_model, inspector_model = get_models()

    if is_image:
        import numpy as np
        nparr = np.frombuffer(image_or_video, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frames = [frame]
        total_frames = 1
        analyzed = 1
    else:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(image_or_video)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 15 != 0:
                continue
            frames.append(frame)
        cap.release()
        os.unlink(tmp_path)
        analyzed = len(frames)

    quality_total = {}
    disease_total = {}
    disease_conf = {}
    inspector_total = {}
    seg_total = {}

    tracked_ids_inspector = set()
    tracked_ids_quality = set()
    tracked_ids_disease = set()

    for frame in frames:
        # 1. 상품감별(inspector) 모드일 때
        if shot_type == 'inspector':
            res = inspector_model.track(frame, conf=0.4, persist=True, verbose=False)[0]
            if res.boxes.id is not None:
                for box, track_id in zip(res.boxes, res.boxes.id):
                    tid = int(track_id)
                    if tid in tracked_ids_inspector:
                        continue
                    tracked_ids_inspector.add(tid)
                    cls = res.names[int(box.cls)]
                    conf = float(box.conf)
                    if cls == 'Discard' and conf < 0.85:
                        cls = 'Ugly'
                    inspector_total[cls] = inspector_total.get(cls, 0) + 1
            else:
                for box in res.boxes:
                    cls = res.names[int(box.cls)]
                    inspector_total[cls] = inspector_total.get(cls, 0) + 1

        # 2. 생산추적(wide, zoom) 모드일 때
        elif shot_type in ('wide', 'zoom'):
            q = quality_model.track(frame, conf=0.5, persist=True, verbose=False)[0]
            if q.boxes.id is not None:
                for box, track_id in zip(q.boxes, q.boxes.id):
                    tid = int(track_id)
                    if tid in tracked_ids_quality:
                        continue
                    tracked_ids_quality.add(tid)
                    cls = q.names[int(box.cls)]
                    quality_total[cls] = quality_total.get(cls, 0) + 1
            else:
                for box in q.boxes:
                    cls = q.names[int(box.cls)]
                    quality_total[cls] = quality_total.get(cls, 0) + 1

        if shot_type == 'zoom':
            d = disease_model.track(frame, conf=0.5, persist=True, verbose=False)[0]
            if d.boxes.id is not None:
                for box, track_id in zip(d.boxes, d.boxes.id):
                    tid = int(track_id)
                    if tid in tracked_ids_disease:
                        continue
                    tracked_ids_disease.add(tid)
                    cls = d.names[int(box.cls)]
                    if cls == 'Healthy':
                        continue
                    disease_total[cls] = disease_total.get(cls, 0) + 1
                    disease_conf.setdefault(cls, []).append(float(box.conf))
            else:
                for box in d.boxes:
                    cls = d.names[int(box.cls)]
                    if cls == 'Healthy':
                        continue
                    disease_total[cls] = disease_total.get(cls, 0) + 1
                    disease_conf.setdefault(cls, []).append(float(box.conf))

            s = seg_model(frame, conf=0.3, verbose=False)[0]
            if s.masks is not None:
                for box, mask in zip(s.boxes, s.masks):
                    cls = s.names[int(box.cls)]
                    area = int(mask.data.sum().item())
                    seg_total.setdefault(cls, []).append(area)

    results = {}
    i_total = sum(inspector_total.values()) or 1
    q_total = sum(quality_total.values()) or 1
    results['inspector'] = [
        {'class_name': k, 'count': v, 'ratio': round(v / i_total * 100, 2)}
        for k, v in inspector_total.items()
    ]
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
    pink_areas = seg_total.get('tom_fruit_pink_poly', [])
    green_areas = seg_total.get('tom_fruit_breaker_poly', [])

    if red_areas:
        red_avg = sum(red_areas) / len(red_areas)
    elif pink_areas:
        # Pink 최대값이 Red의 90%라고 가정
        pink_max = max(pink_areas)
        red_avg = pink_max / 0.9
    elif green_areas:
        # Green 최대값이 Red의 50%라고 가정
        green_max = max(green_areas)
        red_avg = green_max / 0.5
    else:
        red_avg = None
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


def _generate_vision_video(app, session_id, video_bytes, shot_type, output_path, is_image=False):
    with app.app_context():
        try:
            disease_model, quality_model, seg_model, inspector_model = get_models()

            # 색상 및 라벨 설정
            SEG_COLOR = {
                'tom_fruit_breaker_poly': (0, 255, 128),
                'tom_fruit_pink_poly': (147, 20, 255),
                'tom_fruit_red_poly': (0, 0, 255),
            }
            SEG_LABEL = {
                'tom_fruit_breaker_poly': 'Green',
                'tom_fruit_pink_poly': 'Pink',
                'tom_fruit_red_poly': 'Red',
            }
            QUALITY_COLOR = {
                'unripe': (50, 205, 50),
                'half_ripe': (147, 20, 255),
                'ripe': (0, 0, 220),
                'rotten': (60, 60, 60),
                'blossom_end_rot': (30, 30, 150),
                'fruit_cracking': (100, 60, 200),
                'sunscald': (200, 200, 0),
                'brown_rugose': (30, 60, 120),
            }
            INSPECTOR_COLOR = {
                'Premium': (0, 255, 0),  # 녹색
                'Ugly': (0, 165, 255),  # 주황색
                'Discard': (0, 0, 255),  # 빨간색
                'unripe': (255, 255, 0),  # 하늘색
            }

            def draw_text_bg(img, text, pos, scale, color):
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                x, y = pos
                if y - th - 4 < 0: y = y + th + 4
                if x + tw + 4 > img.shape[1]: x = img.shape[1] - tw - 6
                bg = img.copy()
                cv2.rectangle(bg, (x, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
                cv2.addWeighted(bg, 0.5, img, 0.5, 0, img)
                cv2.putText(img, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

            # 통합된 프레임 처리 함수
            def process_frame(frame):
                overlay = frame.copy()

                # A. 상품감별(inspector) 모드 시각화
                if shot_type == 'inspector':
                    res = inspector_model(frame, conf=0.4, verbose=False)[0]
                    for box in res.boxes:
                        cls = res.names[int(box.cls)]
                        conf = float(box.conf)
                        if cls == 'Discard' and conf < 0.85:
                            cls = 'Ugly'
                        color = INSPECTOR_COLOR.get(cls, (200, 200, 200))
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                        draw_text_bg(overlay, f"{cls} {float(box.conf):.2f}", (x1, y1 - 5), 0.4, color)

                # B. 생산추적(wide, zoom) 품질 시각화
                if shot_type in ('wide', 'zoom'):
                    q = quality_model(frame, conf=0.5, verbose=False)[0]
                    for box in q.boxes:
                        cls = q.names[int(box.cls)]
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        color = QUALITY_COLOR.get(cls, (200, 200, 200))
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
                        draw_text_bg(overlay, f"Q:{cls[:4]} {float(box.conf):.2f}", (x1, y2 + 15), 0.3, color)

                # C. 근접 zoom 전용 (질병 + 세그멘테이션)
                if shot_type == 'zoom':
                    # 질병 시각화
                    d = disease_model(frame, conf=0.5, verbose=False)[0]
                    for box in d.boxes:
                        cls = d.names[int(box.cls)]
                        if cls == 'Healthy': continue
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        draw_text_bg(overlay, f"D:{cls[:8]}", (x1, y1 - 8), 0.45, (0, 0, 255))

                    # 세그멘테이션 및 성장도 시각화
                    s = seg_model(frame, conf=0.3, verbose=False)[0]
                    if s.masks is not None:
                        seg_areas = {k: [] for k in SEG_COLOR}
                        for box, mask in zip(s.boxes, s.masks):
                            cls = s.names[int(box.cls)]
                            if cls in seg_areas:
                                seg_areas[cls].append(int(mask.data.sum().item()))

                        red_areas = seg_areas.get('tom_fruit_red_poly', [])
                        red_avg = sum(red_areas) / len(red_areas) if red_areas else None

                        for box, mask in zip(s.boxes, s.masks):
                            cls = s.names[int(box.cls)]
                            color = SEG_COLOR.get(cls, (255, 255, 255))
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                            label = SEG_LABEL.get(cls, cls)
                            area = int(mask.data.sum().item())
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)

                            if red_avg:
                                pct = min(area / red_avg * 100, 100)
                                draw_text_bg(overlay, f"S:{label} {pct:.0f}%", (x1, y1 - 28), 0.5, color)
                                bar_w = x2 - x1
                                cv2.rectangle(overlay, (x1, y2 + 2), (x2, y2 + 10), (50, 50, 50), -1)
                                cv2.rectangle(overlay, (x1, y2 + 2), (x1 + int(bar_w * pct / 100), y2 + 10), color, -1)

                return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

            # 저장 경로 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if is_image:
                # 이미지 처리
                nparr = np.frombuffer(video_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame.shape[1] > 640:
                    ratio = 640 / frame.shape[1]
                    frame = cv2.resize(frame, (640, int(frame.shape[0] * ratio)))
                result = process_frame(frame)
                cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 70])
            else:
                # 영상 처리
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    tmp.write(video_bytes)
                    tmp_path = tmp.name

                cap = cv2.VideoCapture(tmp_path)
                fps = cap.get(cv2.CAP_PROP_FPS) / 2  # 원본의 절반 FPS로 저장

                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                target_w = 640
                target_h = int(orig_h * (target_w / orig_w))

                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (target_w, target_h))

                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frame_count += 1
                    # 2프레임당 1개만 처리하여 인코딩 속도 향상 및 용량 절감
                    if frame_count % 2 != 0: continue

                    frame_resized = cv2.resize(frame, (target_w, target_h))
                    result = process_frame(frame_resized)
                    out.write(result)

                cap.release()
                out.release()
                os.unlink(tmp_path)

            # DB 상태 업데이트
            from sqlalchemy import text
            with db.engine.begin() as conn:
                conn.execute(text("""
                    UPDATE vision_session
                    SET video_status='ready', video_path=:path
                    WHERE session_id=:sid
                """), {'path': output_path, 'sid': session_id})

        except Exception as e:
            traceback.print_exc()
            from sqlalchemy import text
            with db.engine.begin() as conn:
                conn.execute(text("""
                    UPDATE vision_session SET video_status='error' WHERE session_id=:sid
                """), {'sid': session_id})



@bp.route('/')
def index():
    if not g.user:
        return redirect(url_for('auth.login'))

    cult_list = (
        Cultivations.query
        .join(Farms, Cultivations.farm_id == Farms.farm_id)
        .filter(
            Farms.user_id == g.user.user_id,
            Farms.is_active == 'Y'
        )
        .filter(Cultivations.status != 'hidden')
        .all()
    )
    selected_cult = cult_list[0] if cult_list else None

    return render_template('vision.html',
        cult_list=cult_list,
        selected_cult=selected_cult,
        username=g.user.username
    )


@bp.route('/analyze/<int:cult_id>', methods=['POST'])
def analyze(cult_id):
    if not g.user:
        return jsonify({"error": "로그인이 필요합니다"}), 401
    try:
        analysis_mode = request.form.get('analysis_mode', 'farm')  # 'farm' | 'inspector'
        shot_type = request.form.get('shot_type', 'wide') if analysis_mode == 'farm' else 'inspector'

        if 'image' not in request.files:
            return jsonify({"error": "이미지가 없습니다"}), 400

        file = request.files['image']
        image_or_video = file.read()
        is_image = file.content_type.startswith('image/')
        vision_results = _run_vision(image_or_video, shot_type, is_image=is_image)

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

            # inspector 결과 저장
            inspector_list = vision_results.get('inspector', [])
            if inspector_list:
                for i in inspector_list:
                    conn.execute(text("""
                        INSERT INTO vision_inspector (session_id, class_name, count, ratio)
                        VALUES (:sid, :cls, :cnt, :ratio)
                    """), {'sid': session_id, 'cls': i['class_name'], 'cnt': i['count'], 'ratio': i['ratio']})

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

        suffix = '.jpg' if is_image else '.mp4'
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'static', 'vision_output'
        )
        output_path = os.path.join(output_dir, f'vision_{session_id}{suffix}')
        app = current_app._get_current_object()
        t = threading.Thread(target=_generate_vision_video,
                             args=(app, session_id, image_or_video, shot_type, output_path, is_image))
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
            video_url = '/static/vision_output/' + os.path.basename(video_path)
            is_image = video_path.endswith(('.jpg', '.jpeg', '.png'))
            return jsonify({'status': 'ready', 'video_url': video_url, 'is_image': is_image})
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


@bp.route('/history/<int:cult_id>')
def history(cult_id):
    try:
        from sqlalchemy import text
        with db.engine.connect() as conn:
            sessions = conn.execute(text("""
                SELECT s.session_id, s.shot_type, s.analyzed_at, s.zone_name, s.video_path,
                       json_agg(DISTINCT jsonb_build_object(
                           'class_name', q.class_name, 'count', q.count, 'ratio', q.ratio
                       )) FILTER (WHERE q.id IS NOT NULL) as quality,
                       json_agg(DISTINCT jsonb_build_object(
                           'class_name', d.class_name, 'count', d.count, 'avg_conf', d.avg_conf
                       )) FILTER (WHERE d.id IS NOT NULL) as disease,
                       json_agg(DISTINCT jsonb_build_object(
                           'class_name', sg.class_name, 'count', sg.count,
                           'avg_growth', sg.avg_growth
                       )) FILTER (WHERE sg.id IS NOT NULL) as segment,
                       json_agg(DISTINCT jsonb_build_object(
                           'class_name', ins.class_name, 'count', ins.count, 'ratio', ins.ratio
                       )) FILTER (WHERE ins.id IS NOT NULL) as inspector
                FROM vision_session s
                LEFT JOIN vision_quality q ON s.session_id = q.session_id
                LEFT JOIN vision_disease d ON s.session_id = d.session_id
                LEFT JOIN vision_segment sg ON s.session_id = sg.session_id
                LEFT JOIN vision_inspector ins ON s.session_id = ins.session_id
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
                    'video_path': row.video_path or '',
                    'quality': row.quality or [],
                    'disease': row.disease or [],
                    'segment': row.segment or [],
                    'inspector': row.inspector or []
                })

        return jsonify({'cult_id': cult_id, 'history': history}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@bp.route('/session/<int:session_id>', methods=['DELETE'])
def delete_session(session_id):
    try:
        from sqlalchemy import text
        with db.engine.begin() as conn:
            # 영상 파일 삭제
            row = conn.execute(text("""
                SELECT video_path FROM vision_session WHERE session_id=:sid
            """), {'sid': session_id}).fetchone()

            if row and row.video_path and os.path.exists(row.video_path):
                os.remove(row.video_path)

            # DB 삭제
            conn.execute(text("DELETE FROM vision_quality WHERE session_id=:sid"), {'sid': session_id})
            conn.execute(text("DELETE FROM vision_disease WHERE session_id=:sid"), {'sid': session_id})
            conn.execute(text("DELETE FROM vision_segment WHERE session_id=:sid"), {'sid': session_id})
            conn.execute(text("DELETE FROM vision_session WHERE session_id=:sid"), {'sid': session_id})

        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500