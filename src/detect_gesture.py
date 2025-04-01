import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2  # ✅ Sửa đúng tại đây

base_options_gesture = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
gesture_options = vision.GestureRecognizerOptions(
    base_options=base_options_gesture,
    num_hands=2
)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

mp_drawing = mp.solutions.drawing_utils

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = gesture_recognizer.recognize(mp_image)
    return result.gestures, result.handedness, result.hand_landmarks

def get_index_finger_position(hand_landmarks, image_shape):
    if not hand_landmarks:
        return None, None
    h, w, _ = image_shape
    if hasattr(hand_landmarks, 'landmark'):
        landmarks = hand_landmarks.landmark
    else:
        landmarks = hand_landmarks
    index_tip = landmarks[8]
    pixel_x = int((1 - index_tip.x) * w)
    pixel_y = int(index_tip.y * h)
    return pixel_x, pixel_y
