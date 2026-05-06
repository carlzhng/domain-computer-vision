import cv2
import mediapipe as mp
from utils.camerafps import CvFpsCalc
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

hands = mp_hands.Hands(
    static_image_mode =False,
    max_num_hands = 4,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

def get_landmarks(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    landmark_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmark_list.append([lm.x, lm.y, lm.z])
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return landmark_list, frame

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        landmarks, frame = get_landmarks(frame)

        fps = cv_fps_calc.get()
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Domain Expansion Detector!', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()