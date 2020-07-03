# Face-Recognition-in-Camera/Video

# It detects faces from the Camera/Video and recognizes the label(name) of the person. The rule is simple more the images in training more the accuracy in the testing.In training process we give more than 50 images to the recognizer so it can learn the different faces of the same person. For face detection we use haarcascade_frontal_face.xml file to train haarcascade classifier.

# Dataset:
More than 50 images of each person.

# Classifier:
1. LBPH Face Recognizer
2. Haarcascade classifier

# Pre-Requisits:
Must have Python3 installed.
Must have OpenCV module installed.
Must have Numpy module installed.
Must have OS module installed.

# Instructions to run:
1. First clone the project
2. Make your training and testing data.
3. Now first run the faceRecognition.py then run tester.py
4. After training you the model will predict the name of person by fetching face from camera/video.
5. While running the program for second or third time, we can load the trainingData.yml(results of previous training) as a training data.

# Contributors:
> Sharad Kumar Tiwari
