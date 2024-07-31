import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define the data directories
train_dir = 'C:\\Users\\Aryak\\Desktop\\Face detect\\Face detect\\train'
test_dir = 'C:\\Users\\Aryak\\Desktop\\Face detect\\Face detect\\test'

# Get a list of all the emotion class directories
emotion_dirs = os.listdir(train_dir)

# Create empty lists to store the images and labels
X_train = []
y_train = []
X_test = []
y_test = []

# Process the train data
for emotion_dir in emotion_dirs:
    emotion_path = os.path.join(train_dir, emotion_dir)
    image_files = os.listdir(emotion_path)
    for image_file in image_files:
        image_path = os.path.join(emotion_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 48))
        X_train.append(image)
        y_train.append(emotion_dirs.index(emotion_dir))

# Process the test data
for emotion_dir in emotion_dirs:
    emotion_path = os.path.join(test_dir, emotion_dir)
    image_files = os.listdir(emotion_path)
    for image_file in image_files:
        image_path = os.path.join(emotion_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 48))
        X_test.append(image)
        y_test.append(emotion_dirs.index(emotion_dir))

# Convert the lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Save the data to files
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)