# main.py
from detect_gesture import process_frame, gesture_recognizer, mp_drawing, landmark_pb2
from mouse_tool import click_processing, update_click_frame
import threading
import cv2
import time

# Start gesture detection thread
click_thread = threading.Thread(target=click_processing, daemon=True)
click_thread.start()

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    prev_time = time.time()
    target_fps = 30

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