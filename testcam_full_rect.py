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

# Function to detect fire using machine learning-based detection
def detect_fire_ml(frame):
    processed_frame = preprocess_image(frame)
    prediction = model.predict(processed_frame)
    return prediction[0][1] > prediction[0][0]  # Return True if fire is detected

# Function to detect fire using color-based detection
def detect_fire_color(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for detecting fire color
    lower_color = np.array([0, 50, 50])  # Lower bound for red color
    upper_color = np.array([10, 255, 255])  # Upper bound for red color

    # Threshold the HSV image to get only fire color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and check if any contour area is greater than a threshold
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust this threshold based on your specific requirements
            return True  # Fire is detected

    return False  # No fire detected

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Check if fire is detected using color-based detection
    if detect_fire_color(frame):
        # Check if fire is confirmed using machine learning-based detection
        if detect_fire_ml(frame):
            # If both methods confirm fire, draw rectangle and display "Fire Detected" text
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)
            cv2.putText(frame, "Fire Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Fire Detection", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()