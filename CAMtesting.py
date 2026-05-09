import cv2
import mediapipe as mp
import csv
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw
from model.fingerClassifier import KeyPointClassifier

from utils.camerafps import CvFpsCalc
from utils.hand_processing import pre_process_landmark, logging_csv, get_landmarks

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
    mp_draw = mp.solutions.drawing_utils
#-------------------------------------------------------------------------------
    #camera loop
    while True:
        captured,frame = capture.read()

        if not captured:
            print("frame grabbing failed lol")
            break
        
        frame = cv2.flip(frame, 1)
        fps = cv_fps_calc.get()
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(10)
#-------------------------------------------------------------------------------
    #destructor
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()