import cv2
import numpy as np
from keras.models import load_model

classifier = cv2.CascadeClassifier(r"C:\Users\Aryak\Desktop\Face detect\Face detect\haarcascade_frontalface_default.xml")

model = load_model(r"C:\Users\Aryak\Desktop\Face detect\Face detect\final_model.h5")

def get_pred_label(pred):
    labels = ["abhisikta", "adhinayak", "aryak", "bara", "biswabarenya", "Saswat","Sashwat", "smruti ranjan"]
    return labels[pred]

def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(100,100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1,100,100,1)
    img = img/255
    return img

# Open the default camera (usually the first one)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    
    if not ret:  # If frame is not captured properly, break the loop
        break
    
    faces = classifier.detectMultiScale(frame, 1.5, 5)
      
    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
        cv2.putText(frame, get_pred_label(np.argmax(model.predict(preprocess(face)))),
                    (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        
    cv2.imshow("Webcam", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
