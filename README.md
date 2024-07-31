# Face Detection and Recognition System

This project implements a real-time face detection and recognition system using Python, OpenCV, and Keras. The system is capable of detecting faces from a live video feed captured via an IP camera and recognizing them using a pre-trained deep learning model.

## Features

- **Face Detection**: Utilizes Haar Cascade Classifier to detect faces in real-time.
- **Face Recognition**: Uses a Keras model to predict and recognize faces.
- **Data Collection**: Captures and preprocesses face images for training the recognition model.
- **Image Processing**: Applies techniques such as resizing, grayscale conversion, and histogram equalization for improved accuracy.

## Project Structure

- **collect_data.py**: Script to collect face images and save them for training.
- **consolidated_data.py**: Script to preprocess collected images and save them in a format suitable for model training.
- **recognize.py**: Main script to detect and recognize faces from a live video feed.
- **haarcascade_frontalface_default.xml**: Haar Cascade Classifier file for face detection.
- **final_model.h5**: Pre-trained Keras model for face recognition.

## Usage

### 1. Data Collection
Run `collect_data.py` to capture face images.

```bash
python collect_data.py
```
- The script will capture images from the IP camera feed.
- Enter the name of the person being captured when prompted.
- Ensure at least 100 images are captured for each person.

### 2. Preprocess Data
Run consolidated_data.py to preprocess and save the images for training.

```bash
python consolidated_data.py
```
### 3. Run Face Recognition
Run `recognize.py` to start the face detection and recognition system.

```bash
python recognize.py
```
## Note:
- Modify the IP camera URL in the scripts if necessary.
- Ensure the paths to haarcascade_frontalface_default.xml and final_model.h5 are correct.
  
## Dependencies:
- Python 3.x
- OpenCV
- Keras
- NumPy
- Matplotlib
- urllib
- pickle

Install the dependencies using pip:

```bash
pip install opencv-python keras numpy matplotlib
```
## Acknowledgments
The project utilizes the Haar Cascade Classifier for face detection.
Inspired by various online tutorials and resources on face detection and recognition using deep learning.
```bash
python collect_data.py
```
- The script will capture images from the IP camera feed.
- Enter the name of the person being captured when prompted.
- Ensure at least 100 images are captured for each person.
