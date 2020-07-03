import os
import cv2
import numpy as np

def facedetection(test_img):
    gray_image = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    haar_cascade_classifier = cv2.CascadeClassifier('E:/Data Science/PDF/computer vision/FaceRecognition-master/FaceRecognition-master/HaarCascade/haarcascade_frontalface_default.xml')
    face = haar_cascade_classifier.detectMultiScale(gray_image,scaleFactor=1.32,minNeighbors=5)
    
    return face, gray_image
    
def labels_for_training_data(diractory):
    faces=[]
    facesID=[]
    for path,subdirnames,filenames in os.walk(diractory):
        for filename in filenames:
            if filename.startswith('.'):
                print('skipping system file')
                continue
            
            id = os.path.basename(path)
            img_path = os.path.join(path,filename)
            print('img_path = ',img_path)
            print('id = ',id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print('image not loaded properly')
                continue
            face_rect, gray_img = facedetection(test_img)
            if len(face_rect)!=1:
                continue
            (x,y,w,h) = face_rect[0]
            roi_gray = gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            facesID.append(int(id))
            
    return faces,facesID

def train_classifier(faces,facesID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(facesID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,255),3)
    
def put_test(test_img,text,x,y):
    cv2.putText(test_img, text, (x,y), cv2.FONT_HERSHEY_COMPLEX,1, (255,255,0),3)
    