import cv2
import numpy as np
import pyautogui
import threading
import time
from detect import process_frame, get_index_finger_position, draw_landmarks_on_image

# GLOBAL VARIABLES CHO QUÁ TRÌNH CLICK
global_click_frame = None
global_click_lock = threading.Lock()

prev_pixel_x = None
prev_pixel_y = None

# Các biến cho bộ lọc mượt chuột
smoothed_x, smoothed_y = pyautogui.position()  # Khởi tạo vị trí chuột hiện tại

def click_processing():
    """
    Luồng riêng xử lý ảnh flip để nhận diện gesture và di chuyển chuột theo delta có bộ lọc.
    - Nếu không nhận diện được bàn tay, reset delta.
    - Nếu tính ra vị trí mới vượt quá kích thước màn hình, giữ nguyên vị trí chuột hiện tại.
    - Nếu gesture "open_palm" được phát hiện, đặt chuột ngay ở trung tâm màn hình.
    - Nếu gesture "victory" được phát hiện, thực hiện click tại vị trí hiện tại.
    """
    global global_click_frame, prev_pixel_x, prev_pixel_y, smoothed_x, smoothed_y
    alpha = 0.2  # Hệ số lọc (một giá trị nhỏ giúp chuyển động mượt)
    while True:
        frame_to_process = None
        with global_click_lock:
            if global_click_frame is not None:
                frame_to_process = global_click_frame.copy()
        if frame_to_process is None:
            time.sleep(0.01)
            continue

        # Xử lý nhận diện từ frame đã flip
        hand_detection_result, gesture_recognition_result, _ = process_frame(frame_to_process)
        
        # Nếu không phát hiện bàn tay, reset delta và bỏ qua
        if not hand_detection_result.hand_landmarks:
            prev_pixel_x, prev_pixel_y = None, None
            time.sleep(0.01)
            continue

        height, width, _ = frame_to_process.shape
        screen_width, screen_height = pyautogui.size()
        
        # Xử lý từng bàn tay (ở đây chỉ xử lý bàn tay đầu tiên)
        for idx in range(len(hand_detection_result.hand_landmarks)):
            gestures = gesture_recognition_result.gestures[idx]
            if not gestures:
                continue

            top_gesture = gestures[0]
            gesture_name = top_gesture.category_name.lower()
            
            pixel_x, pixel_y = get_index_finger_position(hand_detection_result, frame_to_process.shape)
            if pixel_x is None or pixel_y is None:
                continue

            # Vì frame đã được flip, nếu muốn di chuyển cùng hướng với người dùng, KHÔNG đảo pixel_x.
            # (Nếu cần đảo, thêm: pixel_x = width - pixel_x)
            
            # Nếu tọa độ ngoài phạm vi frame, bỏ qua cập nhật
            if pixel_x < 0 or pixel_x > width or pixel_y < 0 or pixel_y > height:
                continue

            # --- Nếu gesture "open_palm": đặt chuột ở center ---
            if gesture_name == "open_palm":
                center_x = screen_width // 2
                center_y = screen_height // 2
                pyautogui.moveTo(center_x, center_y)
                # Reset delta để tránh tác động lẫn nhau
                prev_pixel_x, prev_pixel_y = pixel_x, pixel_y
                smoothed_x, smoothed_y = center_x, center_y
                continue  # bỏ qua các bước di chuyển delta

            # --- DI CHUYỂN CHUỘT THEO DELTA (relative movement) ---
            if prev_pixel_x is not None and prev_pixel_y is not None:
                dx = pixel_x - prev_pixel_x
                dy = pixel_y - prev_pixel_y

                speed = 5.0  # Hệ số khuếch đại chuyển động chuột
                dx *= speed
                dy *= speed

                # Giới hạn delta (ví dụ: tối đa 500 pixel mỗi lần, bạn có thể điều chỉnh)
                max_delta = 500
                dx = max(min(dx, max_delta), -max_delta)
                dy = max(min(dy, max_delta), -max_delta)

                curr_x, curr_y = pyautogui.position()
                new_x = curr_x + dx
                new_y = curr_y + dy

                # Kiểm tra nếu new_x, new_y ngoài màn hình, giữ nguyên vị trí hiện tại
                if new_x < 0 or new_x >= screen_width or new_y < 0 or new_y >= screen_height:
                    new_x, new_y = curr_x, curr_y

                # Áp dụng bộ lọc smoothing:
                smoothed_x = (1 - alpha) * smoothed_x + alpha * new_x
                smoothed_y = (1 - alpha) * smoothed_y + alpha * new_y

                pyautogui.moveTo(int(smoothed_x), int(smoothed_y))
            else:
                # Nếu chưa có delta, khởi tạo vị trí mượt
                curr_x, curr_y = pyautogui.position()
                smoothed_x, smoothed_y = curr_x, curr_y

            # Cập nhật delta cho lần lặp sau
            prev_pixel_x = pixel_x
            prev_pixel_y = pixel_y

            # --- THỰC HIỆN CLICK NẾU GESTURE LÀ "VICTORY" ---
            if gesture_name == "victory":
                mapped_x = int(pixel_x * screen_width / width)
                mapped_y = int(pixel_y * screen_height / height)
                # Nếu tọa độ vượt ngoài màn hình, giữ nguyên vị trí hiện tại
                if mapped_x < 0 or mapped_x >= screen_width or mapped_y < 0 or mapped_y >= screen_height:
                    mapped_x, mapped_y = pyautogui.position()
                print(f"[CLICK] Gesture: {gesture_name} → Click tại ({mapped_x}, {mapped_y})")
                pyautogui.click(mapped_x, mapped_y)
                time.sleep(0.5)
        time.sleep(0.01)

# Khởi tạo và chạy luồng click (daemon)
click_thread = threading.Thread(target=click_processing, daemon=True)
click_thread.start()

# --- Phần xử lý video chính ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không lấy được frame từ camera.")
        break

    frame = cv2.resize(frame, (640, 480))
    
    curr_time = time.time()
    if curr_time - prev_time < 0.033:  # ~30 fps
        continue
    prev_time = curr_time

    # Ảnh dùng để hiển thị (không flip)
    display_frame = frame.copy()
    
    # Tạo phiên bản ảnh flip để xử lý click (theo góc nhìn người dùng)
    flipped_frame = cv2.flip(frame, 1)
    with global_click_lock:
        global_click_frame = flipped_frame.copy()

    # Xử lý nhận diện cho hiển thị
    _, _, annotated_image = process_frame(display_frame)
    annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    cv2.imshow("Hand Gesture Recognition", annotated_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
