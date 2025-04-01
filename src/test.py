import cv2
import time
import mediapipe as mp
import numpy as np
import threading
import pyautogui
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# --- BUFFER TRUNG BÌNH / EMA ---
buffer_size = 7
pos_x_buffer = deque(maxlen=buffer_size)
pos_y_buffer = deque(maxlen=buffer_size)

# --- CẤU HÌNH LỌC MƯỢT ---
filter_mode = "ema"  # Chọn giữa: "mean", "ema", "none"
ema_alpha = 0.2       # Hệ số cho EMA (chỉ dùng nếu filter_mode == "ema")
ema_x = None
ema_y = None

# --- GLOBAL CONFIGURATION ---
global_click_lock = threading.Lock()
global_click_frame = None

prev_pixel_x = None
prev_pixel_y = None
smoothed_x, smoothed_y = pyautogui.position()

speed = 7.0
sensitivity = 1
click_cooldown = 0.5
last_click_time = 0

# Cấu hình mượt chuột & FPS
delta_threshold = 2
target_fps = 30

# --- CẤU HÌNH MÔ HÌNH GESTURE RECOGNIZER ---
base_options_gesture = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
gesture_options = vision.GestureRecognizerOptions(
    base_options=base_options_gesture,
    num_hands=2
)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

# --- Công cụ vẽ ---
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

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

def click_processing():
    global global_click_frame, prev_pixel_x, prev_pixel_y, last_click_time
    global smoothed_x, smoothed_y, ema_x, ema_y

    last_frame_time = time.time()

    mouse_locked = False  # Khóa chuột khi tay trái làm Victory

    while True:
        now = time.time()
        dt = now - last_frame_time
        if dt < 1 / target_fps:
            time.sleep(1 / target_fps - dt)
        last_frame_time = time.time()

        frame_to_process = None
        with global_click_lock:
            if global_click_frame is not None:
                frame_to_process = global_click_frame.copy()
        if frame_to_process is None:
            continue

        gestures_result, handedness_result, hand_landmarks_result = process_frame(frame_to_process)
        if not hand_landmarks_result:
            prev_pixel_x, prev_pixel_y = None, None
            continue

        height, width, _ = frame_to_process.shape
        screen_width, screen_height = pyautogui.size()
        right_hand_found = False

        # --- Kiểm tra xem tay trái có đang "Victory" hay không ---
        mouse_locked = False
        for i in range(len(hand_landmarks_result)):
            handed_label = handedness_result[i][0].category_name
            gesture = gestures_result[i][0].category_name.lower() if gestures_result[i] else None
            if handed_label == "Left" and gesture == "victory":
                mouse_locked = True
                break

        for i in range(len(hand_landmarks_result)):
            handed_label = handedness_result[i][0].category_name
            gesture = gestures_result[i][0].category_name.lower() if gestures_result[i] else None

            # Tay trái điều khiển hành động
            if handed_label == "Left" and gesture:
                if gesture == "open_palm":
                    center_x = screen_width // 2
                    center_y = screen_height // 2
                    pyautogui.moveTo(center_x, center_y)
                    prev_pixel_x, prev_pixel_y = None, None
                    smoothed_x, smoothed_y = center_x, center_y
                    pos_x_buffer.clear()
                    pos_y_buffer.clear()
                    ema_x, ema_y = center_x, center_y
                elif gesture == "closed_fist":
                    current_time = time.time()
                    if current_time - last_click_time > click_cooldown:
                        pyautogui.click()
                        last_click_time = current_time
                elif gesture == "thumb_up":
                    pyautogui.scroll(100)
                elif gesture == "thumb_down":
                    pyautogui.scroll(-100)
                # ✅ KHÔNG cần gán mouse_locked ở đây nữa

            # Tay phải điều khiển chuột (nếu không bị lock)
            if handed_label == "Right" and not mouse_locked:
                right_hand_found = True
                pixel_x, pixel_y = get_index_finger_position(hand_landmarks_result[i], frame_to_process.shape)
                if pixel_x is None or pixel_y is None:
                    continue

                if prev_pixel_x is None or prev_pixel_y is None:
                    prev_pixel_x, prev_pixel_y = pixel_x, pixel_y
                    curr_x, curr_y = pyautogui.position()
                    smoothed_x, smoothed_y = curr_x, curr_y
                    pos_x_buffer.clear()
                    pos_y_buffer.clear()
                    pos_x_buffer.append(curr_x)
                    pos_y_buffer.append(curr_y)
                    ema_x, ema_y = curr_x, curr_y
                else:
                    dx = (pixel_x - prev_pixel_x) * speed * sensitivity
                    dy = (pixel_y - prev_pixel_y) * speed * sensitivity
                    curr_x, curr_y = pyautogui.position()
                    new_x = max(0, min(curr_x + dx, screen_width))
                    new_y = max(0, min(curr_y + dy, screen_height))

                    # --- Áp dụng bộ lọc ---
                    if filter_mode == "mean":
                        pos_x_buffer.append(new_x)
                        pos_y_buffer.append(new_y)
                        smoothed_x = sum(pos_x_buffer) / len(pos_x_buffer)
                        smoothed_y = sum(pos_y_buffer) / len(pos_y_buffer)

                    elif filter_mode == "ema":
                        ema_x = ema_alpha * new_x + (1 - ema_alpha) * ema_x
                        ema_y = ema_alpha * new_y + (1 - ema_alpha) * ema_y
                        smoothed_x, smoothed_y = ema_x, ema_y

                    else:  # "none"
                        smoothed_x, smoothed_y = new_x, new_y

                    if abs(curr_x - smoothed_x) > delta_threshold or abs(curr_y - smoothed_y) > delta_threshold:
                        pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

                prev_pixel_x = pixel_x
                prev_pixel_y = pixel_y

        if not right_hand_found:
            prev_pixel_x, prev_pixel_y = None, None

def update_click_frame(frame):
    global global_click_frame
    with global_click_lock:
        global_click_frame = frame.copy()

click_thread = threading.Thread(target=click_processing, daemon=True)
click_thread.start()

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không lấy được frame từ camera.")
            break

        flipped_frame = frame  # Không flip hình

        curr_time = time.time()
        if curr_time - prev_time < 1 / target_fps:
            continue
        prev_time = curr_time

        update_click_frame(flipped_frame)

        gestures_result, handedness_result, hand_landmarks_result = process_frame(flipped_frame)
        annotated_image = flipped_frame.copy()

        for i in range(len(gestures_result)):
            if gestures_result[i]:
                print(f"[DEBUG] Tay {i} - Gesture: {gestures_result[i][0].category_name}")

        if hand_landmarks_result:
            for i in range(len(hand_landmarks_result)):
                landmarks = hand_landmarks_result[i]
                if not hasattr(landmarks, 'landmark'):
                    temp_landmarks = landmark_pb2.NormalizedLandmarkList()
                    for lm in landmarks:
                        new_lm = temp_landmarks.landmark.add()
                        new_lm.x = lm.x
                        new_lm.y = lm.y
                        new_lm.z = lm.z
                        if hasattr(lm, "visibility"):
                            new_lm.visibility = lm.visibility
                        if hasattr(lm, "presence"):
                            new_lm.presence = lm.presence
                else:
                    temp_landmarks = landmarks
                mp_drawing.draw_landmarks(
                    annotated_image,
                    temp_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(thickness=2)
                )

        cv2.imshow("Hand Gesture Recognition", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
