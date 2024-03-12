import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model("fire_detection_model.h5")

# Function to preprocess an image
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))  # Resize image to match input size
    img = img / 255.0  # Normalize pixel values
    return img.reshape(-1, 128, 128, 3)  # Reshape and return

# Function to detect fire in a frame
def detect_fire(frame):
    processed_frame = preprocess_image(frame)
    prediction = model.predict(processed_frame)
    if prediction[0][1] > prediction[0][0]:  # If fire is detected
        return True
    else:
        return False

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Check if fire is detected
    if detect_fire(frame):
        # Overlay text "Fire Detected" on the frame
        cv2.putText(frame, "Fire Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Fire Detection", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()