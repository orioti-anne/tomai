from ultralytics import YOLO
import cv2
import os

DL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(DL_DIR, 'models')
VIDEO_DIR = os.path.join(DL_DIR, 'video')
OUTPUT_DIR = os.path.join(DL_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("모델 로딩 중...")
disease_model = YOLO(os.path.join(MODEL_DIR, 'disease_best.pt'))
quality_model = YOLO(os.path.join(MODEL_DIR, 'quality_best.pt'))
seg_model = YOLO(os.path.join(MODEL_DIR, 'seg_best.pt'))
print("모델 로딩 완료!")

VIDEO_NAME = 'zone2_wide.mp4'
VIDEO = os.path.join(VIDEO_DIR, VIDEO_NAME)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, f'full_result_{VIDEO_NAME}'),
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (w, h)
)

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
    'unripe':          (0, 200, 0),
    'half_ripe':       (0, 200, 255),
    'ripe':            (0, 80, 255),
    'rotten':          (120, 120, 120),
    'mold':            (80, 80, 200),
    'anthracnose':     (200, 80, 80),
    'blossom_end_rot': (200, 150, 0),
    'fruit_cracking':  (180, 100, 50),
    'sunscald':        (0, 220, 220),
    'brown_rugose':    (100, 60, 180),
}

def draw_text_with_bg(img, text, pos, font_scale, color, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = pos
    if y - th - 6 < 0:
        y = y + th + 6
    if x + tw + 4 > img.shape[1]:
        x = img.shape[1] - tw - 6
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    d_result = disease_model(frame, conf=0.5, verbose=False)[0]
    q_result = quality_model(frame, conf=0.5, verbose=False)[0]
    s_result = seg_model(frame, conf=0.4, verbose=False)[0]

    overlay = frame.copy()

    # ── 질병 박스 ──
    for box in d_result.boxes:
        cls_name = d_result.names[int(box.cls)]
        conf = float(box.conf)
        if cls_name == 'Healthy':
            continue  # Healthy는 표시 안함
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        draw_text_with_bg(overlay, f"D:{cls_name} {conf:.2f}", (x1, y1 - 8), 0.45, (0, 0, 255))

    # ── 품질 박스 ──
    for box in q_result.boxes:
        cls_name = q_result.names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        color = QUALITY_COLOR.get(cls_name, (200, 200, 200))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        draw_text_with_bg(overlay, f"Q:{cls_name} {conf:.2f}", (x1, y2 + 15), 0.45, color)

    # ── 세그 박스 + 성장률 ──
    size_by_class = {k: [] for k in SEG_COLOR}
    detections = []

    if s_result.masks is not None:
        for box, mask in zip(s_result.boxes, s_result.masks):
            cls_name = s_result.names[int(box.cls)]
            area = int(mask.data.sum().item())
            detections.append({
                'cls': cls_name,
                'area': area,
                'xyxy': box.xyxy[0].tolist(),
            })
            if cls_name in size_by_class:
                size_by_class[cls_name].append(area)

    red_areas = size_by_class['tom_fruit_red_poly']
    red_avg = sum(red_areas) / len(red_areas) if red_areas else None

    for det in detections:
        cls = det['cls']
        color = SEG_COLOR.get(cls, (255, 255, 255))
        x1, y1, x2, y2 = [int(v) for v in det['xyxy']]
        label = SEG_LABEL.get(cls, cls)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)

        if red_avg and red_avg > 0:
            pct = min(det['area'] / red_avg * 100, 100)
            draw_text_with_bg(overlay, f"S:{label} {pct:.0f}%", (x1, y1 - 28), 0.55, color)
            bar_w = x2 - x1
            bar_filled = int(bar_w * pct / 100)
            cv2.rectangle(overlay, (x1, y2 + 12), (x2, y2 + 22), (50, 50, 50), -1)
            cv2.rectangle(overlay, (x1, y2 + 12), (x1 + bar_filled, y2 + 22), color, -1)
        else:
            draw_text_with_bg(overlay, f"S:{label}", (x1, y1 - 28), 0.55, color)

    result = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    out.write(result)

    if frame_count % 30 == 0:
        print(f"프레임 {frame_count} | 질병:{len(d_result.boxes)} 품질:{len(q_result.boxes)} 세그:{len(detections)}")

cap.release()
out.release()
print(f"\n완료! → {OUTPUT_DIR}/full_result_{VIDEO_NAME}")