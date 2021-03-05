import cv2 as cv
import face_recognition
import os
import numpy as np
#####################################################################
#                       Initialization
#####################################################################
pathDirectory = 'Training Images'
images = []
person = 'Rohit'
#####################################################################
#                       TRAINING IMAGES
#####################################################################
my_list = os.listdir(pathDirectory)
for img in os.listdir(pathDirectory + '/' + person):
        curImg = cv.imread(pathDirectory + '/' + person + '/' + img)
        images.append(curImg)


def find_encodings(all_images):
    encode_list = []
    for img1 in all_images:
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img1)[0]
        encode_list.append(encode)
    return encode_list


encodings = find_encodings(images)
print('Encoding Process is Successfully Completed')
#####################################################################
#                       TESTING IMAGE
#####################################################################
# imgTest = face_recognition.load_image_file('Test/three 1.jpg')
imgTest = face_recognition.load_image_file('Test/three 2.jpg')
# imgTest = face_recognition.load_image_file('Test/three 3.jpg')
# imgTest = face_recognition.load_image_file('Test/two.jpg')
# imgTest = face_recognition.load_image_file('Test/two 2.jpg')

#####################################################################
imgTest = cv.resize(imgTest, (0, 0), None, 2, 2)
imgTest = cv.cvtColor(imgTest, cv.COLOR_BGR2RGB)

#####################################################################
#             Locating faces and Encoding in test image
#####################################################################
face_locations_test = face_recognition.face_locations(imgTest)
encode_face_test = face_recognition.face_encodings(imgTest)

#####################################################################
#####################################################################
for encode, face in zip(encode_face_test, face_locations_test):
    #####################################################################
    #                   Comparing the facial encoding
    #####################################################################
    comparison = face_recognition.compare_faces(encodings, encode, tolerance=0.5)
    #####################################################################
    #            Facial Distance between input and final image
    #####################################################################
    face_distance = face_recognition.face_distance(encodings, encode)
    #####################################################################
    matchIndex = np.argmin(np.array(face_distance))
    #####################################################################
    if comparison[matchIndex]:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    #####################################################################
    #####################################################################
    cv.rectangle(imgTest, (face[3], face[0]), (face[1], face[2]), color, 2)
    cv.putText(imgTest, f'{comparison[matchIndex]} {round(face_distance[matchIndex], 2)}', (face[3], face[0]),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    #####################################################################
cv.imshow('Test', imgTest)
cv.waitKey(0)
