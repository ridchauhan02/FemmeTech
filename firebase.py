import cv2
import os
import pyrebase
import time

# Firebase Configuration
config = {
    'apiKey': "AIzaSyC9E_q_-w-_0vhknPjAilTsJKjbQK03EuA",
  'authDomain': "attendancesystem-a0a33.firebaseapp.com",
  "databaseURL": "http://attendancesystem-a0a33.firebaseapp.com",
  'projectId': "attendancesystem-a0a33",
  'storageBucket': "attendancesystem-a0a33.appspot.com",
  'messagingSenderId': "350409091675",
  'appId': "1:350409091675:web:6b7668f42e756b5ff9ed7e",
  'measurementId': "G-QTPG3ZPP7R"
    
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
db = firebase.database()

# Create a directory to store the captured images
output_directory = 'captured_faces'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for default camera, change it if you have multiple cameras

# Initialize the face detection model (you may need to install a suitable face detection model)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the attendance count from Firebase Realtime Database
attendance_count = db.child("attendance").get().val() or 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Capture and store the face when a face is detected
    if len(faces) > 0:
        attendance_count += 1
        image_filename = os.path.join(output_directory, f'face_{attendance_count}.jpg')
        cv2.imwrite(image_filename, frame)

        # Upload the captured face to Firebase Storage
        storage.child(f'faces/face_{attendance_count}.jpg').put(image_filename)

        # Update attendance count in Firebase Realtime Database
        db.child("attendance").set(attendance_count)

    cv2.imshow('Face Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
