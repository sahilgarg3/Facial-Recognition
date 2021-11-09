# Facial-Recognition

## [Face Recognition in Images](ImgFaceFinder.py)

### Requirements
- OpenCV
- NumPy
- [Face Recognition](https://pypi.org/project/face-recognition/)
- OS
---
### User Initializations
- Path of the Directory (contains images of each person)
  
  e.g.: [`Training Images`](https://github.com/sahilgarg3/Facial-Recognition/tree/main/Training%20Images) contains folder with persons' name
- Name of the Person

  e.g.:[`Rohit`](https://github.com/sahilgarg3/Facial-Recognition/tree/main/Training%20Images/Rohit) contains multiple images of the person

---
### Importing Images
```
my_list = os.listdir(pathDirectory)
for img in os.listdir(pathDirectory + '/' + person):
        curImg = cv.imread(pathDirectory + '/' + person + '/' + img)
        images.append(curImg)
```
The above code will Import all the images of the person and store them in the `images` list.

### Training Images
`find_encodings` fuction take the list of images as the input and **return** the `list of facial encodings`
```
print('Encoding Process is Successfully Completed')
```
This print statement is the indicator which indicated the completion of encoding process.

### Testing
```
# imgTest = face_recognition.load_image_file('Test/three 1.jpg')
imgTest = face_recognition.load_image_file('Test/three 2.jpg')
# imgTest = face_recognition.load_image_file('Test/three 3.jpg')
# imgTest = face_recognition.load_image_file('Test/two.jpg')
# imgTest = face_recognition.load_image_file('Test/two 2.jpg')
```
Directory named [`Test`](https://github.com/sahilgarg3/Facial-Recognition/tree/main/Test) contains 6 images, try any

The matched face/faces will have green bounding box around the face whereas the un-matcherd face/faces will have red bounding box around the face. 

> ### Virat ![Facial Recognition 1](https://user-images.githubusercontent.com/79501547/140869800-d6bd8dab-3b2a-4e75-9189-c98b582e84e0.png)


> ### Rohit ![Facial Recognition 2](https://user-images.githubusercontent.com/79501547/140869818-53778014-f4c4-482a-a8a2-1eb4faabb773.png)


## [Attendance System](AttendenceSystem.py)

### Requirements
- OpenCV
- NumPy
- [Face Recognition](https://pypi.org/project/face-recognition/)
- OS
- Datetime
---
### User Initializations
- Path of the Directory (contains images of each person)
  
  e.g.: [`Training Images`](https://github.com/sahilgarg3/Facial-Recognition/tree/main/Training%20Images) contains folder with persons' name
---

1. Insert the image/images of the individuals to find their facial encodings.
    - Similar to earlier one
2. Through webcam or any number of camera attached, detected all of the faces.
3. Every detected face will be compared to the encodings of the faces inserted in step one.
4. Recognized faces will retun the name of the person and Unrecognized faces will be return Unknown.
5. Each and Every unique face will be inserted in the Attendence.csv file, as the Name of the individual and the time at which his/her face recognized. 
