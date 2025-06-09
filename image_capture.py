import cv2
import os

"""
This script is to capture images using the webcam
"""

def capture_photo():
    # Define the directory where the image will be saved 
    save_dir = "image_2"
    i = 500
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Open a connection to the default webcam (device 0)
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press SPACE to capture the photo, or ESC to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Display the live video feed
        cv2.imshow("Webcam", frame)

        # Wait for a key press for 1 millisecond
        key = cv2.waitKey(1)
        # ESC key pressed (ASCII code 27)
        if key % 256 == 27:
            print("Escape hit, closing...")
            break
        # SPACE key pressed (ASCII code 32)
        elif key % 256 == 32:
            # Build the full file path for saving the image
            img_path = os.path.join(save_dir, f"data_{i}.png")
            cv2.imwrite(img_path, frame)
            print(f"Photo saved as {img_path}")
            i+=1

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_photo()
