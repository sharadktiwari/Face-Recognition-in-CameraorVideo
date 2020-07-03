import os
import numpy as np
import cv2
import FaceRecognition as fr

#faces,facesID =fr.labels_for_training_data('E:/Data Science/PDF/computer vision/FaceRecognition-master/FaceRecognition-master/trainingImages')
#face_recognizer = fr.train_classifier(faces, facesID)
#face_recognizer.write('E:/Data Science/PDF/computer vision/FaceRecognition-master/FaceRecognition-master/Face Recognition in Camera/trainingData.yml')

#Uncomment below line for subsequent runs
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('E:/Data Science/PDF/computer vision/FaceRecognition-master/FaceRecognition-master/Face Recognition in Camera/trainingData.yml')#use this to load training data for subsequent runs

name = {0:'Prianka',1:'Kangna',2:'Mahesh',3:'SHARAD'}

cap = cv2.VideoCapture(0) #for camera
cap = cv2.VideoCapture('address') #for video

while 1:
    ret,frame = cap.read()
    faces_detected,gray_img=fr.facedetection(frame)
    print('faces detected',faces_detected)

    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray = gray_img[y:y+h,x:x+w]
        label,confidence=face_recognizer.predict(roi_gray)
        print('confidence = ',confidence)
        print('label = ',label)
        fr.draw_rect(frame, face)
        predicted_name = name[label]
        
        fr.put_test(frame, predicted_name, x, y)
        
        resized_image = cv2.resize(frame,(900,800))
        cv2.imshow('predicted image',resized_image)
        key = cv2.waitKey(1)
        if key==ord('q'):
            break
    key = cv2.waitKey(1)
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


