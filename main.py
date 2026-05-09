import cv2
import mediapipe as mp
import csv
from utils.camerafps import CvFpsCalc
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw
from model.fingerClassifier import KeyPointClassifier


# Initializing MediaPipe - Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

#processes landmark data to be relative to the wrist
def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0][0], landmark_list[0][1]
    temp_landmark_list = []

    for lp in landmark_list:
        temp_landmark_list.append(lp[0] - base_x)
        temp_landmark_list.append(lp[1] - base_y)

    return temp_landmark_list

#data logger; appends cordinate data to a csv file with the corresponding label
def logging_csv(label, landmark_list):
    with open('data/fingerCoords.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, *landmark_list])

#function to overlay hand landmarks on the video feed
def get_landmarks(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    all_hands_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            current_hand_points = []
            for lm in hand_landmarks.landmark:
                current_hand_points.append([lm.x, lm.y, lm.z])

            all_hands_list.append(current_hand_points)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return all_hands_list, frame

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    keypoint_classifier = KeyPointClassifier()
    labels = ["Unlimited Void", "Neutral"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        landmarks, frame = get_landmarks(frame)

        if landmarks:
            for hand in landmarks:
                processed_hand = pre_process_landmark(hand)
                hand_sign_id = keypoint_classifier(processed_hand)
                sign_name = labels[hand_sign_id]

                wrist_x = int(hand[0][0] * frame.shape[1])
                wrist_y = int(hand[0][1] * frame.shape[0])

                cv2.putText(frame, sign_name, (wrist_x, wrist_y -20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

        fps = cv_fps_calc.get()
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Domain Expansion Detector!', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('k'):
            if landmarks:
                for hand in landmarks:
                    processed_data = pre_process_landmark(hand)
                    logging_csv(1, processed_data)
                    print("Logged Neutral data point")
            else:
                print("No hand in frame to record")
        
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()