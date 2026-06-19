from ultralytics import YOLO
import os

# 확인하신 경로로 설정
model_dir = 'smartfarm/dl/models'
model_names = ['disease_best.pt', 'inspector_best.pt', 'quality_best.pt', 'seg_best.pt']

for name in model_names:
    path = os.path.join(model_dir, name)
    if os.path.exists(path):
        print(f"\n--- 🚀 {name} 변환 시작 ---")
        try:
            model = YOLO(path)
            # nms=True: 후처리를 하드웨어 가속에 포함
            # int8=True: 모델 용량을 줄이고 뉴럴 엔진(ANE) 사용 최적화
            model.export(format='coreml', nms=True, int8=True)
            print(f"✅ {name} 변환 성공!")
        except Exception as e:
            print(f"❌ {name} 변환 중 오류 발생: {e}")
    else:
        print(f"⚠️ 파일을 찾을 수 없습니다: {path}")

print("\n✨ 모든 작업이 완료되었습니다!")
