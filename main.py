import cv2
import csv
import mediapipe as mp
from utils.camerafps import CvFpsCalc
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw
from model.fingerClassifier import KeyPointClassifier

from utils.camerafps import CvFpsCalc
from utils.hand_processing import pre_process_landmark, log_csv, get_landmarks, labels

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

    #basic instructions
    print("\n"+ "=" * 10 + "< Hand Sign Data Collection >" + "=" * 10)
    print("INSTRUCTIONS:")
    print("Press 'q' to quit the program.")
    print("Press 'n' to cycle forward through labels.")
    print("Press 'b' to cycle backward through labels.")
    print("Press 's' to save the current hand coordinates with the selected label.")
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
        if landmarks:
            for hand in landmarks:
                rel_hand_cords = pre_process_landmark(hand)
                wrist_x = int(hand[0][0] * frame.shape[1])
                wrist_y = int(hand[0][1] * frame.shape[0])

                #identifying hand sign
                hand_sign_id = keypoint_classifier(rel_hand_cords)
                sign_name = labels[hand_sign_id]

                cv2.putText(frame, sign_name, (wrist_x, wrist_y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
                
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
        if key == ord('q'):
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

        # Save Data (Press 's')
        elif key == ord('s'):
            if landmarks:
                for hand in landmarks:
                    processed_data = pre_process_landmark(hand)
                    log_csv(current_label, processed_data, 'data/fingerCoords.csv')
                
                # Detailed print statement
                print(f"LOGGED: ID {current_label} | Sign: {labels[current_label]}")
            else:
                print("No hand in frame to record!")
#-------------------------------------------------------------------------------
    #destructor
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()