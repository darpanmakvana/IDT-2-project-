# from sklearn.neighbors import KNeighborsClassifier
# import cv2
# import pickle
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime


# from win32com.client import Dispatch

# def speak(str1):
#     speak=Dispatch(("SAPI.SpVoice"))
#     speak.Speak(str1)

# video=cv2.VideoCapture(0)
# facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# with open('data/names.pkl', 'rb') as w:
#     LABELS=pickle.load(w)
# with open('data/faces_data.pkl', 'rb') as f:
#     FACES=pickle.load(f)

# print('Shape of Faces matrix --> ', FACES.shape)

# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# imgBackground=cv2.imread("background.png")

# COL_NAMES = ['NAME', 'TIME']

# while True:
#     ret,frame=video.read()
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces=facedetect.detectMultiScale(gray, 1.3 ,5)
#     for (x,y,w,h) in faces:
#         crop_img=frame[y:y+h, x:x+w, :]
#         resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
#         output=knn.predict(resized_img)
#         ts=time.time()
#         date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
#         exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
#         cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
#         cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
#         attendance=[str(output[0]), str(timestamp)]
#     imgBackground[162:162 + 480, 55:55 + 640] = frame
#     cv2.imshow("Frame",imgBackground)
#     k=cv2.waitKey(1)
#     if k==ord('o'):
#         speak("Attendance Taken..")
#         time.sleep(5)
#         if exist:
#             with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                 writer=csv.writer(csvfile)
#                 writer.writerow(attendance)
#             csvfile.close()
#         else:
#             with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                 writer=csv.writer(csvfile)
#                 writer.writerow(COL_NAMES)
#                 writer.writerow(attendance)
#             csvfile.close()
#     if k==ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()

from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import pyttsx3  # Importing pyttsx3 for text-to-speech functionality

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load labels and face data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)
    

# Debugging prints to check shapes
print('Shape of Faces matrix --> ', FACES.shape)
print('Shape of Labels --> ', len(LABELS))

# Ensure FACES and LABELS have the same number of samples
if FACES.shape[0] != len(LABELS):
    min_length = min(FACES.shape[0], len(LABELS))
    FACES = FACES[:min_length]
    LABELS = LABELS[:min_length]
    print("Adjusted shapes to match:")
    print(f"FACES shape: {FACES.shape}, LABELS length: {len(LABELS)}")

# Create and fit KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Attendance column names
COL_NAMES = ['NAME', 'TIME']

# Get current date
date = datetime.now().strftime("%d-%m-%Y")

# Main loop for capturing video
while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        
        # Draw rectangles and labels on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        attendance = [str(output[0]), str(timestamp)]
    
    # Display the frame with detected faces
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)

    # Handle attendance recording
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
        # Write attendance to CSV
        with open("Attendance/Attendance_" + date + ".csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)  # Write headers if the file is new
            writer.writerow(attendance)  # Write the attendance record
    if k == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()

