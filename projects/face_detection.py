'''
Created on Apr 28, 2019

@author: dsj529

implementation of the viola-jones face detection algorithm as packaged in the OpenCV library
'''
import os, os.path

import cv2

BASE_PATH = '../data/face_detect'
CASCADE_CLASSIFIER_PATH = os.path.join(BASE_PATH, 'haarcascade_frontalface_alt.xml')
cascadeClassifier = cv2.CascadeClassifier(CASCADE_CLASSIFIER_PATH)

for i in range(3):
    image = cv2.imread(os.path.join(BASE_PATH, 'image{}.jpg'.format(i+1)))
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detectedFaces = cascadeClassifier.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    for(x,y, width, height) in detectedFaces:
        cv2.rectangle(image, (x, y), (x+width, y+height), (0,0,255), 10)
    
    cv2.imwrite(os.path.join(BASE_PATH, 'result{}.jpg'.format(i+1)), image)
    