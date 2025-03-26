import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker

# # Cấu hình mô hình
# base_options = mp.tasks.BaseOptions(model_asset_path='code/hand_landmarker.task')
# options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
# detector = vision.HandLandmarker.create_from_options(options)

# # Mở ảnh và chạy nhận diện bàn tay
# image = mp.Image.create_from_file("code/temp_image.jpg")
# result = detector.detect(image)

# # Hiển thị kết quả
# for hand in result.hand_landmarks:
#     print("Bàn tay được phát hiện:")
#     for i, landmark in enumerate(hand):
#         print(f"  - Điểm {i}: ({landmark.x}, {landmark.y}, {landmark.z})")


# --- Mở kết nối với camera ---
import cv2

cap = cv2.VideoCapture(0)  # 0 là camera mặc định, thay đổi nếu có nhiều camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không lấy được frame từ camera.")
        break
    
    frame = cv2.flip(frame, 1)
    # Chuyển frame từ BGR (OpenCV) sang RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tạo đối tượng mp.Image từ numpy array
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    # Áp dụng mô hình phát hiện landmark bàn tay
    
    # Vẽ landmark lên ảnh

    # Chuyển về BGR để hiển thị bằng OpenCV
    # rgb_frame = cv2.cvtColor(mp_image, cv2.COLOR_RGB2BGR)

    # Hiển thị stream video
    cv2.imshow("Hand Landmarks", mp_image)

    # Nhấn 'q' để thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(type(mp_image))

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()