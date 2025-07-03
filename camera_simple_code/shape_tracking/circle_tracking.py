#로지텍 카메라로 보이는 물체 YOLO로 추론
from ultralytics import YOLO
import cv2

# 학습된 모델 로드
model = YOLO("best.pt")

# Camera 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추론
    results = model.predict(source=frame, conf=0.5, save=False, show=False)

    # 결과를 OpenCV용으로 시각화
    annotated_frame = results[0].plot()

    # circle이 어느 방향에 있는지 출력하도록 함
    boxes = results[0].boxes

    found_circle = False

    if boxes:  # 객체가 탐지되었을 경우
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:
            # 중심 x좌표 계산
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = (x1 + x2) / 2

                frame_width = frame.shape[1]

                if center_x < frame_width * 1/3:
                    print("left")
                elif center_x > frame_width * 2/3:
                    print("right")
                else:
                    print("center")
                found_circle = True
                break

    # 창으로 표시
    cv2.imshow("YOLO Webcam", annotated_frame)

    # q 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
