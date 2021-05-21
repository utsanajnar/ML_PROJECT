# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# object detection webcam example
# usage: python object_detection_webcam.py

# right now YOLOv3 is being used for detecting objects.
# It's a heavy model to run on CPU. You might see the latency
# in output frames.
# To-Do: Add tiny YOLO for real time object detection

# import necessary packages
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import numpy as np
from django.shortcuts import render, HttpResponse
# open webcam


def Run(request):
    webcam = cv2.VideoCapture("asdf.mp4")




    
    if not webcam.isOpened():
        print("Could not open webcam")
        return render(request,"object_list.html")




    frame_width = int(webcam.get(3)) 
    frame_height = int(webcam.get(4)) 
    
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('static/output_list.mp4',  
                            cv2.VideoWriter_fourcc(*'MP4V'), 
                            20, size) 



    answer =[]
# loop through frames
    while webcam.isOpened():
        print("Video On")
# read frame from webcam 
        for i in range(10):
            status, frame = webcam.read()

        if not status:
    #  print("Could not read frame")
            print(np.unique(np.array(answer)))
            context={}
            count=0
            for item in np.unique(np.array(answer)):
                context[count]= item
                count=count+1
            print("Video Ended...")
            webcam.release()
            result.release()
            cv2.destroyAllWindows()
            return render(request,"object_list.html",{'context':context})

# apply object detection
        bbox, label, conf = cv.detect_common_objects(frame,0.8,0.5)
        for k in label:
            answer.append(k)
        print(bbox, label, conf)

# draw bounding box over detected objects
        out = draw_bbox(frame, bbox, label, conf)

# display output
        cv2.imshow("Real-time object detection", out)
        result.write(out)
# press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(np.unique(np.array(answer)))
            break

# release resources
    webcam.release()
    result.release()
    cv2.destroyAllWindows()        
    return render(request,"object_list.html")