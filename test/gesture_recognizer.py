import cv2 
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --- Khởi tạo đối tượng GestureRecognizer ---
base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# --- Thiết lập vẽ landmark ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands  # Để lấy thông tin HAND_CONNECTIONS

# --- Mở kết nối với camera ---
cap = cv2.VideoCapture(0)  # 0 là camera mặc định

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không lấy được frame từ camera")
        break

    # Chuyển từ BGR (OpenCV) sang RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tạo đối tượng mp.Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # --- Áp dụng mô hình Gesture Recognizer ---
    recognition_result = recognizer.recognize(mp_image)

    # Tạo bản sao của frame để vẽ kết quả
    annotated_image = frame.copy()

    # Kiểm tra nếu có bàn tay được nhận diện
    if recognition_result.gestures:
        for i, (gesture_list, hand_landmarks) in enumerate(zip(recognition_result.gestures, recognition_result.hand_landmarks)):
            top_gesture = gesture_list[0]  # Lấy gesture có độ chính xác cao nhất của mỗi bàn tay

            # Chuyển các landmark về dạng protobuf để sử dụng hàm vẽ của MediaPipe
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_landmarks
            ])

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Hiển thị tên gesture và score lên màn hình
            text = f"Hand {i+1}: {top_gesture.category_name} ({top_gesture.score:.2f})"
            cv2.putText(annotated_image, text, (10, 30 + i * 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

    # Hiển thị kết quả
    cv2.imshow("Gesture Recognition", annotated_image)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
