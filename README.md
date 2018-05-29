# Exam_Notes_Detection

At the time of exams most of the time student share their notes via social media and after the exam gets over it become really difficut to delete all those images manually. So face this problem I have created this system which detects exam notes(pictures which are clicked from mobile camera) and deletes it.


## Getting Started

System contains 4 python files, 1 trained model file and 2 folders. 
I have used LeNet Convolutional Neural Networks and Keras.

### Prerequisites
```
Keras
Scikit
cv2
matplotlib
imutils
```

### Installing
I have already trained the model so you can direcly use it.
```
mymodel.h5
```
You can also train you own network by running "train_network.py"
to train you own network you have to add images to images/notes and images/not_notes (There were many of my personal images so I did not upload it)
```
run: train_network.py
```
Testing the network. Add testing images to "examples/" and to run test.py give command line argument like below:
```
python test_network.py --image examples/image_name.jpg
```
To Delete the image from the desired folder change the path from delete_images.py and directly run the file
```
python delete_images.py
```

## Built With

* [Keras](https://keras.io/) - Deep Leanring framework
* [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf/) - Research Paper

## Examples
<img src="https://github.com/chanduparmar/notes_detection/blob/master/test1.JPG">
<img src="https://github.com/chanduparmar/notes_detection/blob/master/test2.JPG">

