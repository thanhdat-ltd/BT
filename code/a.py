import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Các thiết lập vẽ landmark ---
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Tạo đối tượng landmark protobuf để vẽ
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks
        ])

        # Vẽ landmark và các đường nối
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Tính toán vị trí vẽ nhãn (handedness)
        height, width, _ = annotated_image.shape
        x_coordinates = [lm.x for lm in hand_landmarks]
        y_coordinates = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Vẽ nhãn bên trái hoặc bên phải
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

# --- Khởi tạo mô hình HandLandmarker ---
base_options = python.BaseOptions(model_asset_path='code\hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# --- Mở kết nối với camera ---
cap = cv2.VideoCapture(0)  # 0 là camera mặc định, thay đổi nếu có nhiều camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không lấy được frame từ camera.")
        break

    # Chuyển frame từ BGR (OpenCV) sang RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tạo đối tượng mp.Image từ numpy array
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Áp dụng mô hình phát hiện landmark bàn tay
    detection_result = detector.detect(mp_image)

    # Vẽ landmark lên ảnh
    annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)

    # Chuyển về BGR để hiển thị bằng OpenCV
    annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Hiển thị stream video
    cv2.imshow("Hand Landmarks", annotated_bgr)

    # Nhấn 'q' để thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
