import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import numpy as np
from django.shortcuts import render, HttpResponse
# open webcam

def Run():
    webcam = cv2.VideoCapture("video1.mp4")
    while webcam.isOpened():
        status, frame = webcam.read()
        
        if not status:
            print("Could not read frame")
            return 
        cv2.imshow("Video-Run", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
# release resources
    webcam.release()
    cv2.destroyAllWindows()
    return 
Run()