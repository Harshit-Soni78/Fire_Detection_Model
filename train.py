# Import necessary libraries
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define parameters
fire_dir = "fire_dataset\\fire_images"
non_fire_dir = "fire_dataset\\non_fire_images"
img_size = (128, 128)

# Load and preprocess the data
def load_data(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        try:
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:  # Check if image is successfully loaded
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
            else:
                print(f"Failed to load image: {os.path.join(directory, filename)}")
        except Exception as e:
            print(f"Error loading image: {os.path.join(directory, filename)} - {e}")
    return images, labels


fire_images, fire_labels = load_data(fire_dir, 1)
non_fire_images, non_fire_labels = load_data(non_fire_dir, 0)

X = np.array(fire_images + non_fire_images) / 255.0
y = np.array(fire_labels + non_fire_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save("fire_detection_model.h5")
print("Model saved successfully!")
