# detect.py
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CẤU HÌNH CHO VIỆC VẼ LANDMARK ---
MARGIN = 10          # pixels, để hiển thị text cách mép ảnh
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # màu xanh lá nổi bật

# --- KHỞI TẠO MÔ HÌNH GESTURE RECOGNIZER ---
base_options_gesture = python.BaseOptions(model_asset_path='models\\gesture_recognizer.task' )
gesture_options = vision.GestureRecognizerOptions(base_options=base_options_gesture)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

# --- KHỞI TẠO MÔ HÌNH HAND LANDMARKER ---
base_options_hand = python.BaseOptions(model_asset_path='models\\hand_landmarker.task')
hand_options = vision.HandLandmarkerOptions(base_options=base_options_hand, num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

def draw_landmarks_on_image(rgb_image, detection_result, gesture_result):
    """
    Vẽ landmark, nối các điểm, hiển thị thông tin gesture và
    vẽ biểu tượng (hình tròn màu đỏ) tại landmark id 8 của bàn tay đầu tiên.
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        top_gesture = gesture_result.gestures[idx][0] if gesture_result.gestures and len(gesture_result.gestures) > idx else None

        # Tạo đối tượng protobuf landmark để vẽ
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks
        ])

        # Vẽ landmark và các đường nối
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Lấy kích thước ảnh để tính vị trí pixel
        height, width, _ = annotated_image.shape
        x_coordinates = [lm.x for lm in hand_landmarks]
        y_coordinates = [lm.y for lm in hand_landmarks]
        
        # --- LẤY VỊ TRÍ LANDMARK ID = 8 (đầu ngón trỏ) ---
        index_finger_tip = hand_landmarks[8]
        pixel_x = int(index_finger_tip.x * width)
        pixel_y = int(index_finger_tip.y * height)
        # Vẽ biểu tượng con trỏ (hình tròn màu đỏ, bán kính 10 pixels)
        if idx == 0:
            cv2.circle(annotated_image, (pixel_x, pixel_y), 10, (255, 0, 0), -1)

        # Vị trí để vẽ nhãn (dựa theo điểm ngoài cùng bên trái)
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        text = f"{handedness[0].category_name}"
        if top_gesture:
            text += f" - {top_gesture.category_name} ({top_gesture.score:.2f})"
        cv2.putText(annotated_image, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE,
                    HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def process_frame(frame):
    """
    Nhận một frame (BGR), chuyển đổi, chạy nhận diện hand và gesture,
    và trả về (hand_detection_result, gesture_recognition_result, annotated_image).
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    hand_detection_result = hand_detector.detect(mp_image)
    gesture_recognition_result = gesture_recognizer.recognize(mp_image)
    annotated_image = draw_landmarks_on_image(rgb_frame, hand_detection_result, gesture_recognition_result)
    return hand_detection_result, gesture_recognition_result, annotated_image

def get_index_finger_position(detection_result, image_shape):
    """
    Lấy vị trí pixel của landmark index finger tip (id 8) từ kết quả detection.
    image_shape: (height, width, channels)
    """
    hand_landmarks_list = detection_result.hand_landmarks
    if not hand_landmarks_list:
        return None, None
    height, width, _ = image_shape
    hand_landmarks = hand_landmarks_list[0]
    index_finger_tip = hand_landmarks[8]
    pixel_x = int(index_finger_tip.x * width)
    pixel_y = int(index_finger_tip.y * height)
    return pixel_x, pixel_y
