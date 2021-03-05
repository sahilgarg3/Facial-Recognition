import cv2 as cv
import face_recognition

#####################################################################
#                       TRAINING IMAGE
#####################################################################
train_img = face_recognition.load_image_file('Training Images/Rohit/1.jpg')
train_img = cv.resize(train_img, (0, 0), None, 2, 2)
train_img = cv.cvtColor(train_img, cv.COLOR_BGR2RGB)

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
#                   Locating faces and Encoding
#####################################################################
face_locations = face_recognition.face_locations(train_img)[0]          # order y1, x2, y2, x1
# First element as it contain only one person in the image
cv.rectangle(train_img, (face_locations[3], face_locations[0]), (face_locations[1], face_locations[2]), (255, 0, 255),
             2)
encoded_face = face_recognition.face_encodings(train_img)[0]

#####################################################################
#             Locating faces and Encoding in test image
#####################################################################
face_locations_test = face_recognition.face_locations(imgTest)
encode_face_test = face_recognition.face_encodings(imgTest)

#####################################################################
for face, encode in zip(face_locations_test, encode_face_test):
    #####################################################################
    #                   Comparing the facial encoding
    #####################################################################
    comparison = face_recognition.compare_faces([encoded_face], encode, tolerance=0.5)

    #####################################################################
    #            Facial Distance between input and final image
    #####################################################################
    face_distance = face_recognition.face_distance([encoded_face], encode)

    #####################################################################
    #####################################################################
    if comparison[0]:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    #####################################################################
    #####################################################################
    cv.rectangle(imgTest, (face[3], face[0]), (face[1], face[2]), color, 2)
    cv.putText(imgTest, f'{comparison} {round(face_distance[0], 2)}', (face[3], face[0]), cv.FONT_HERSHEY_SIMPLEX, 0.6,
               color, 2)
    #####################################################################
cv.imshow('Train', train_img)
cv.imshow('Test', imgTest)
cv.waitKey(0)
