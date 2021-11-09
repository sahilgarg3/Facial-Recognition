import cv2 as cv
import face_recognition
import os
import numpy as np
from datetime import datetime
#####################################################################
#                       INITIALIZATION
#####################################################################
pathDirectory = 'Training Images'
images = []
class_names = []
#####################################################################
#                   Import Images
#####################################################################
my_list = os.listdir(pathDirectory)
for person in my_list:
    for img in os.listdir(pathDirectory + '/' + person):
        curImg = cv.imread(pathDirectory + '/' + person + '/' + img)
        images.append(curImg)
        class_names.append(person)
#####################################################################
# print(len(images))
# print(len(class_names))
#####################################################################
#                       Encoding
#####################################################################


def find_encodings(all_images):
    encode_list = []
    location_list = []
    for img1 in all_images:
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
        location = face_recognition.face_locations(img1)[0]
        encode = face_recognition.face_encodings(img1)[0]
        encode_list.append(encode)
        location_list.append(location)
    return encode_list
#####################################################################
#                   MARKING ATTENDANCE
#####################################################################


def attendance(person):
    with open('Attendance.csv', 'r+') as f:
        details = f.readlines()
        person_reached = []
        for line in details:
            entry = line.split(', ')
            person_reached.append(entry[0])
        if person not in person_reached:
            now = datetime.now()
            text = now.strftime("%H:%M:%S")
            f.writelines(f'\n{person}, {text}')
#####################################################################
#               For Capturing Screen Rather than Webcam
#####################################################################


# def capture_screen(bbox=(300, 300, 690+300, 530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv.cvtColor(capScr, cv.COLOR_BGR2GRAY)
#     return capScr
#####################################################################
#                      Encoding List
#####################################################################
encodings = find_encodings(images)
print('Encoding Process is Successfully Completed')
#####################################################################
cap = cv.VideoCapture(0)
#####################################################################


while True:
    _, img = cap.read()
    # img = capture_screen()
    img_resize = cv.resize(img, (0, 0), None, 0.25, 0.25)
    img_resize = cv.cvtColor(img_resize, cv.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(img_resize)
    encodeCurFrame = face_recognition.face_encodings(img_resize, facesCurFrame)

    for encode_face, face_loc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodings, encode_face, tolerance=0.5)
        face_dis = face_recognition.face_distance(encodings, encode_face)
        matchIndex = np.argmin(np.array(face_dis))

        if face_dis[matchIndex] < 0.50:                 # Or        # if matches[matchIndex]:
            name = class_names[matchIndex].upper()
        else:
            name = 'UnKnown'
        attendance(name)
        y1, x2, y2, x1 = face_loc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), -1)
        cv.putText(img, name, (x1 + 5, y2 - 5), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv.imshow('Capture', img)
    cv.waitKey(1)
