import cv2
import csv
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

#-------------------------------------------------------------------------------
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