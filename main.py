import cv2
import csv
import mediapipe as mp
import time
import os
import subprocess
import threading 

from utils.camerafps import CvFpsCalc
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw
from model.fingerClassifier import KeyPointClassifier
from utils.camerafps import CvFpsCalc
from utils.hand_processing import pre_process_landmark, log_csv, get_landmarks, labels


ps_process = subprocess.Popen(
    ['powershell.exe', '-NoExit', '-Command', '-'],
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

def press_key(key):
    cmd = f'Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait("{key}")\n'
    ps_process.stdin.write(cmd.encode())
    ps_process.stdin.flush()

#-------------------------------------------------------------------------------
def main():  
    #capture window setup
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    #fps calculator setup
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    #hands setup
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    #intialize classifier and labels
    keypoint_classifier = KeyPointClassifier()
    current_label = 0
    COOLDOWN = 0.1
    last_press_time = 0


    #basic instructions
    print("\n"+ "=" * 10 + "< Hand Sign Data Collection >" + "=" * 10)
    print("INSTRUCTIONS:")
    print("Press 'p' to quit the program.")
    print("Press 'n' to cycle forward through labels.")
    print("Press 'b' to cycle backward through labels.")
    print("Press 'l' to save the current hand coordinates with the selected label.")
    print("="*49+"\n")
#-------------------------------------------------------------------------------
    #main camera loop
    while True:
    #basic frame capture and flip
        captured,frame = capture.read()

        if not captured:
            print("frame grabbing failed lol")
            break
        frame = cv2.flip(frame, 1)

    #hand landmark processing
        landmarks, frame = get_landmarks(frame)
        dual_hand_data = [0] * 84 

        if landmarks:
            if len(landmarks) == 2:
                # Fill both slots
                dual_hand_data[0:42] = pre_process_landmark(landmarks[0])
                dual_hand_data[42:84] = pre_process_landmark(landmarks[1])
            else:
                # Only one hand: Decide if it's the "Left" or "Right" slot based on X position
                wrist_x = landmarks[0][0][0]
                processed = pre_process_landmark(landmarks[0])
                
                if wrist_x < 0.5:
                    dual_hand_data[0:42] = processed  # Put in Left slot
                else:
                    dual_hand_data[42:84] = processed # Put in Right slot

            hand_sign_id = keypoint_classifier(dual_hand_data)
            sign_name = labels[hand_sign_id]

            wrist_x = int(landmarks[0][0][0] * frame.shape[1])
            wrist_y = int(landmarks[0][0][1] * frame.shape[0])
            cv2.putText(frame, sign_name, (wrist_x, wrist_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
            if sign_name != "neutral":
                now = time.time()
                if now - last_press_time > COOLDOWN:
                    threading.Thread(target=press_key, args=('g',), daemon=True).start()
                    last_press_time = now
            
    #framerate display
        fps = cv_fps_calc.get()
        cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    
        #point logging and program termination
        cv2.putText(frame, 
                    f"RECORDING MODE: {current_label} ({labels[current_label] if current_label < len(labels) else 'Unknown'})", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0))
        
        #show frame
        cv2.imshow('camera', frame)
            
        key = cv2.waitKey(1) & 0xFF

        # 1. Quit Program
        if key == ord('p'):
            break
        elif key == ord('n'):
            current_label += 1
            if current_label >= len(labels):
                current_label = 0 # Loop back to the start
            
            print(f"ID: {current_label} | Selected Sign: {labels[current_label]}")

        # Toggle Label DOWN (Press 'b')
        elif key == ord('b'):
            current_label -= 1
            if current_label < 0:
                current_label = len(labels) - 1 # Loop to the end
            print(f"ID: {current_label} | Selected Sign: {labels[current_label]}")

        # Save Data (Press 'l')
        elif key == ord('l'):
                if landmarks:
                    # SAVE THE 84-POINT DATA, NOT INDIVIDUAL HANDS
                    log_csv(current_label, dual_hand_data, 'data/fingercords.csv')
                    print(f"LOGGED 84 POINTS: ID {current_label} | Sign: {labels[current_label]}")
                else:
                    print("No hand in frame to record!")

#-------------------------------------------------------------------------------
    #destructor
    ps_process.stdin.close()
    ps_process.terminate()
    capture.release()
    cv2.destroyAllWindows()
    os.system('stty sane')

if __name__ == "__main__":
    main()