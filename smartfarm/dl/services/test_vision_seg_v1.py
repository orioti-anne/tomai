from ultralytics import YOLO
import cv2
import os
import numpy as np

DL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(DL_DIR, 'models')
VIDEO_DIR = os.path.join(DL_DIR, 'video')
OUTPUT_DIR = os.path.join(DL_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("모델 로딩 중...")
seg_model = YOLO(os.path.join(MODEL_DIR, 'seg_best.pt'))
print("모델 로딩 완료!")

VIDEO_NAME = 'zone1_zoom.mp4'
VIDEO = os.path.join(VIDEO_DIR, VIDEO_NAME)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, f'growth_rate_{VIDEO_NAME}'),
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (w, h)
)

COLOR = {
    'tom_fruit_breaker_poly': (0, 165, 255),
    'tom_fruit_pink_poly':    (147, 20, 255),
    'tom_fruit_red_poly':     (0, 0, 255),
}

LABEL_MAP = {
    'tom_fruit_breaker_poly': 'Breaker',
    'tom_fruit_pink_poly':    'Pink',
    'tom_fruit_red_poly':     'Red',
}

def draw_text_with_bg(img, text, pos, font_scale, color, thickness=2):
    """텍스트 배경 포함 그리기"""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = pos
    # 화면 밖으로 나가면 안쪽으로
    if y - th - 6 < 0:
        y = y + th + 6
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    s_result = seg_model(frame, conf=0.2, imgsz=1280, verbose=False)[0]

    if s_result.masks is None:
        out.write(frame)
        continue

    detections = []
    size_by_class = {k: [] for k in COLOR}

    for box, mask in zip(s_result.boxes, s_result.masks):
        cls_name = s_result.names[int(box.cls)]
        area = int(mask.data.sum().item())
        detections.append({
            'cls': cls_name,
            'area': area,
            'xywh': box.xywh[0].tolist(),
            'xyxy': box.xyxy[0].tolist(),
            'conf': float(box.conf)
        })
        if cls_name in size_by_class:
            size_by_class[cls_name].append(area)

    # 완숙 평균 = 기준 100%
    red_areas = size_by_class['tom_fruit_red_poly']
    red_avg = sum(red_areas) / len(red_areas) if red_areas else None

    overlay = frame.copy()

    for det in detections:
        cls = det['cls']
        color = COLOR.get(cls, (255, 255, 255))
        x1, y1, x2, y2 = [int(v) for v in det['xyxy']]

        # 박스
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = LABEL_MAP.get(cls, cls)

        if red_avg and red_avg > 0:
            pct = min(det['area'] / red_avg * 100, 100)
            text = f"{label} {pct:.0f}%"

            # 텍스트 (배경 포함)
            draw_text_with_bg(overlay, text, (x1, y1 - 8), 0.55, color)

            # 성장률 바
            bar_w = x2 - x1
            bar_filled = int(bar_w * pct / 100)
            cv2.rectangle(overlay, (x1, y2 + 2), (x2, y2 + 10), (50, 50, 50), -1)
            cv2.rectangle(overlay, (x1, y2 + 2), (x1 + bar_filled, y2 + 10), color, -1)
        else:
            draw_text_with_bg(overlay, label, (x1, y1 - 8), 0.55, color)

    # 우상단 요약 패널
    panel_x = w - 220
    cv2.rectangle(overlay, (panel_x - 8, 8), (w - 8, 28 + len([k for k, v in size_by_class.items() if v]) * 28), (0, 0, 0), -1)
    summary_y = 28
    for cls, areas in size_by_class.items():
        if areas:
            avg = sum(areas) / len(areas)
            pct = (avg / red_avg * 100) if red_avg else 0
            pct = min(pct, 100)
            color = COLOR[cls]
            label = LABEL_MAP[cls]
            summary = f"{label}: {len(areas)}개  {pct:.0f}%"
            cv2.putText(overlay, summary, (panel_x, summary_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            summary_y += 28

    result = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    out.write(result)

    if frame_count % 30 == 0:
        print(f"프레임 {frame_count} 처리 중...")

cap.release()
out.release()
print(f"\n완료! → {OUTPUT_DIR}/growth_rate_{VIDEO_NAME}")