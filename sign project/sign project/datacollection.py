import cv2
import os
import numpy as np

# -------------------- SETTINGS --------------------
DATA_DIR = "data"
GESTURE_NAME = "A"     # jis sign ka data lena hai (A/B/C/Hello etc)
TOTAL_IMAGES = 200     # jitni images capture karni hai
# ---------------------------------------------------

# Folder create
gesture_path = os.path.join(DATA_DIR, GESTURE_NAME)
os.makedirs(gesture_path, exist_ok=True)

# For DroidCam and many webcams, CAP_DSHOW is more stable on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera. Please check DroidCam connection.")
        exit()

print(f"Collecting data for gesture: {GESTURE_NAME}")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV — skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin color range
    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        # Draw box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

        # Crop hand
        hand = frame[y:y+h, x:x+w]

        if hand.size > 0:
            hand = cv2.resize(hand, (200, 200))

            # Save image
            img_path = os.path.join(gesture_path, f"{count}.jpg")
            cv2.imwrite(img_path, hand)
            count += 1

            cv2.putText(frame, f"Saved: {count}/{TOTAL_IMAGES}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if count >= TOTAL_IMAGES:
        print("Data collection completed!")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
