import os
import cv2
from utils.hand_processing import pre_process_landmark, log_csv, get_landmarks

def batch_process_images(folder_path):

    # Map the filename strings to dataset IDs
    label_map = {
        "gojo": 0,       # Matches "Unlimited Void"
        "neutral": 1     # Matches "Neutral"
    }
    # Verify folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The directory '{folder_path}' does not exist!")
        return
    
    # Gather all images from the directory
    supported_extensions = ('.jpg', '.jpeg', '.png')
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if f.lower().endswith(supported_extensions)]

    print("=" * 60)
    print(f"Starting batch extraction on {len(image_files)} images...")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for idx, filename in enumerate(image_files):
        # Determine label based on file name string checking
        filename_lower = filename.lower()
        current_label_id = None
        
        for keyword, label_id in label_map.items():
            if keyword in filename_lower:
                current_label_id = label_id
                break

        # If image does not match any keyword, skip it
        if current_label_id is None:
            print(f"[{idx+1}/{len(image_files)}] Skipped: '{filename}' (No matching label keyword found)")
            continue

        # Load the static image
        image_path = os.path.join(folder_path, filename)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"[{idx+1}/{len(image_files)}] Error: Could not read image '{filename}'")
            continue

        # Extract landmarks using your custom wrapper
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

if __name__ == "__main__":
    # Create an images directory or reference where you downloaded the dataset
    IMAGE_DIR = "./downloaded_dataset"
    batch_process_images(IMAGE_DIR)