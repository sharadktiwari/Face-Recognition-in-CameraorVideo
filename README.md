# Face-Recognition-in-Camera/Video

# It detects faces from the Camera/Video and recognizes the label(name) of the person.
 The rule is simple more the images in training more the accuracy in the testing.
 In the training process, we give more than 50 images to the recognizer so it can learn the different facets of the same person.
 For face detection, we use haarcascade_frontal_face.xml file to train haar cascade classifier.

# Dataset:
More than 50 images of each person.

# Classifier:
1. LBPH Face Recognizer
2. Haar cascade classifier

# Pre-Requisites:
Must have Python3 installed.
Must have OpenCV module installed.
Must have Numpy module installed.
Must have OS module installed.

# Instructions to run:
1. First clone the project
2. Make your training and testing data. Make different sub-directories with the label name for each person. 
   Specify the name of the person with directory label in your program.
3. Now first run the faceRecognition.py then run tester.py
4. After training the model will predict the name of the person by fetching face from camera/video.
5. While running the program for the second or third time, we can load the trainingData.yml(results of previous training) as training data.

# Contributors:
> Sharad Kumar Tiwari
