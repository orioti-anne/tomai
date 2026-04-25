# smartfarm/dl/services/test_vision.py
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

VIDEO_NAME = 'zone1_zoom.mp4'
VIDEO = os.path.join(VIDEO_DIR, VIDEO_NAME)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 출력 영상 설정 (3개 모델 나란히)
out = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, f'result_{VIDEO_NAME}'),
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (w * 3, h)
)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 각 모델 추론
    d_result = disease_model(frame, conf=0.3, verbose=False)[0]
    q_result = quality_model(frame, conf=0.3, verbose=False)[0]
    s_result = seg_model(frame, conf=0.3, verbose=False)[0]

    # 각 모델 결과 시각화
    d_frame = d_result.plot()
    q_frame = q_result.plot()
    s_frame = s_result.plot()

    # 상단에 모델명 텍스트
    cv2.putText(d_frame, f'Disease | frame:{frame_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(q_frame, f'Quality | {len(q_result.boxes)}개', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(s_frame, f'Segment | {len(s_result.boxes)}개', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 3개 나란히 합치기
    combined = cv2.hconcat([d_frame, q_frame, s_frame])
    out.write(combined)

    # 30프레임마다 로그
    if frame_count % 30 == 0:
        print(f"프레임 {frame_count} | 질병:{len(d_result.boxes)} 품질:{len(q_result.boxes)} 세그:{len(s_result.boxes)}")

cap.release()
out.release()
print(f"\n완료! 결과 저장: {OUTPUT_DIR}/result_{VIDEO_NAME}")