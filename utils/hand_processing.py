import cv2
import csv
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

#-------------------------------------------------------------------------------
#label list
labels = ["neutral", "nue", "toad", "wolf", "gojo", "sukuna"]

# Map the filename strings to dataset IDs
label_map = {
    "neutral":0,
    "nue":1,
    "toad": 2,
    "wolf":3,
    "gojo": 4,
    "sukuna": 5
}
#-------------------------------------------------------------------------------

#hand cordinate normalization
def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0][0], landmark_list[0][1]
    temp_landmark_list = []

    for lp in landmark_list:
        temp_landmark_list.append(lp[0] - base_x)
        temp_landmark_list.append(lp[1] - base_y)

    return temp_landmark_list

#-------------------------------------------------------------------------------
#data logger; appends cordinate data to a csv file with the corresponding label
def log_csv(label, landmark_list, filename):
    with open(filename, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, *landmark_list])

#-------------------------------------------------------------------------------

def get_landmarks(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    all_hands_list = []

    if results.multi_hand_landmarks:
        # Collect hands
        for hand_landmarks in results.multi_hand_landmarks:
            current_hand_points = []
            for lm in hand_landmarks.landmark:
                current_hand_points.append([lm.x, lm.y, lm.z])
            all_hands_list.append(current_hand_points)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 2. SORT hands by X-coordinate (Left to Right)
        # This ensures Hand 0 is always the one further to the left on screen
        all_hands_list.sort(key=lambda x: x[0][0]) 

    return all_hands_list, frame

#-------------------------------------------------------------------------------