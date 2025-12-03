import cv2
import numpy as np
import os

# Path to the dataset folder
dataset_path = 'dataset'

# Create subfolders for dummy Aadhaar IDs
aadhaar_ids = ['123456789012', '987654321098', '112233445566']
for aadhaar_id in aadhaar_ids:
    folder_path = os.path.join(dataset_path, aadhaar_id)
    os.makedirs(folder_path, exist_ok=True)

    # Create dummy face images
    for i in range(5):  # Create 5 images per person
        # Create a black image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Draw a "face" (circle as face)
        cv2.circle(img, (100, 100), 60, (255, 255, 255), -1)  # White circle (face)
        
        # Draw eyes (small white circles)
        cv2.circle(img, (70, 80), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(img, (130, 80), 10, (0, 0, 0), -1)  # Right eye
        
        # Draw a mouth (black ellipse)
        cv2.ellipse(img, (100, 130), (30, 15), 0, 0, 180, (0, 0, 0), -1)  # Mouth
        
        # Draw a nose (small white circle)
        cv2.circle(img, (100, 110), 5, (0, 0, 0), -1)  # Nose
        
        # Add some text to indicate the Aadhaar ID
        cv2.putText(img, f"ID: {aadhaar_id}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save the image
        img_path = os.path.join(folder_path, f"img{i + 1}.jpg")
        cv2.imwrite(img_path, img)

print("Dummy faces created.")
