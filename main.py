import cv2
# Import the class from your utils folder
from utils.camerafps import CvFpsCalc

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # 1. Initialize the FPS calculator
    # buffer_len=10 means it averages the last 10 frames
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # 2. Get the smoothed FPS value
        fps = cv_fps_calc.get()

        # 3. Display the FPS on the frame
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Domain Expansion Detector!', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()