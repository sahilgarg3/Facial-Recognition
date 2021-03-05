# Facial-Recognition

## [Face Recognition in Images](ImgFaceFinder.py)
1. Insert the name of the individual to be recognized. 
2. Insert new image to find the same individual out of the n number of people in the new inserted image. 
3. The matched face will have green bounding box around the face. 
4. The un-matcherd face/faces will have red bounding box around the face. 


## [Face Recognition through Webcam](main.py)
1. Insert the image/images of the individuals to find their respective encodings.
2. Through webcam, the n number of faces detected.
3. Every detected face will be compared to the encodings of the faces inserted in step one.
4. Unrecognized faces will be taken as Unknown 
5. Each and Every unique face will be inserted in the Attendence.csv file, as the Name of the individual and the time at which his/her face recognized. 
