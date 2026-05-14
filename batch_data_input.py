import os
import cv2
from utils.hand_processing import pre_process_landmark, log_csv, get_landmarks, label_map

def batch_process_images(folder_path):
#-------------------------------------------------------------------------------
#verifcations before processing

    # Verify folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The directory '{folder_path}' does not exist!")
        return
    
    # Gather all images from the directory
    supported_extensions = ('.jpg', '.jpeg', '.png')
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if f.lower().endswith(supported_extensions)]

    print("=" * 50)
    print(f"Starting batch extraction on {len(image_files)} images...")
    print("=" * 50)

    success_count = 0
    fail_count = 0
#-------------------------------------------------------------------------------
#extraction loop

    #extracting corrisponding label based on file name
    for idx, filename in enumerate(image_files):
        filename_lower = filename.lower()
        
        # Split file name by underscore and take the first element [0]
        first_word = filename_lower.split('_')[0]
        
        # Direct dictionary lookup to get the label ID
        current_label_id = label_map.get(first_word)

        if current_label_id is None:
            print(f"[{idx+1}/{len(image_files)}] Skipped: '{filename}' (No match for '{first_word}')")
            continue

        # Load the static image
        image_path = os.path.join(folder_path, filename)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"[{idx+1}/{len(image_files)}] Error: Could not read image '{filename}'")
            continue

        # Extract landmarks
        landmarks, _ = get_landmarks(frame)

        if landmarks:
            # Process and log data points exactly like your camera loop does
            for hand in landmarks:
                processed_data = pre_process_landmark(hand)
                log_csv(current_label_id, processed_data, 'data/fingercords_V2.csv')
            
            print(f"[{idx+1}/{len(image_files)}] LOGGED: '{filename}' -> Label ID: {current_label_id}")
            success_count += 1
        else:
            print(f"[{idx+1}/{len(image_files)}] FAILED: No hands detected in '{filename}'")
            fail_count += 1

    print("=" * 60)
    print(" EXTRACTION COMPLETE")
    print(f" Successfully processed: {success_count} images")
    print(f" Failed (MediaPipe missed hand): {fail_count} images")
    print("=" * 60)

#-------------------------------------------------------------------------------
#main

if __name__ == "__main__":
    IMAGE_DIR = "data/images"
    
    # Check if it actually exists to be safe
    if not os.path.exists(IMAGE_DIR):
        print(f"I'm at {os.getcwd()}, but I can't find {IMAGE_DIR}")
    
    batch_process_images(IMAGE_DIR)