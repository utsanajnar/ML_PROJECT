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
import os
import numpy as np
from django.shortcuts import render, HttpResponse
import datetime
# open webcam


initialize = True
net = None
dest_dir = os.path.expanduser('~') + os.path.sep + '.cvlib' + os.path.sep + 'object_detection' + os.path.sep + 'yolo' + os.path.sep + 'yolov3'
classes = None
COLORS = np.random.uniform(0, 255, size=(80, 3))




def populate_class_labels():

    class_file_name = 'yolov3_classes.txt'
    class_file_abs_path = dest_dir + os.path.sep + class_file_name
    url = 'https://github.com/arunponnusamy/object-detection-opencv/raw/master/yolov3.txt'
    if not os.path.exists(class_file_abs_path):
        download_file(url=url, file_name=class_file_name, dest_dir=dest_dir)
    f = open(class_file_abs_path, 'r')
    classes = [line.strip() for line in f.readlines()]

    return classes




def detect_common_object(image,inputlabel, confidence=0.5, nms_thresh=0.3, model='yolov3', enable_gpu=False):

    Height, Width = image.shape[:2]
    scale = 0.00392

    global classes
    global dest_dir

    if model == 'yolov3-tiny':
        config_file_name = 'yolov3-tiny.cfg'
        cfg_url = "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg"
        weights_file_name = 'yolov3-tiny.weights'
        weights_url = 'https://pjreddie.com/media/files/yolov3-tiny.weights'
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)


    else:
        config_file_name = 'yolov3.cfg'
        cfg_url = 'https://github.com/arunponnusamy/object-detection-opencv/raw/master/yolov3.cfg'
        weights_file_name = 'yolov3.weights'
        weights_url = 'https://pjreddie.com/media/files/yolov3.weights'
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)    

    config_file_abs_path = dest_dir + os.path.sep + config_file_name
    weights_file_abs_path = dest_dir + os.path.sep + weights_file_name    
    
    if not os.path.exists(config_file_abs_path):
        download_file(url=cfg_url, file_name=config_file_name, dest_dir=dest_dir)

    if not os.path.exists(weights_file_abs_path):
        download_file(url=weights_url, file_name=weights_file_name, dest_dir=dest_dir)    

    global initialize
    global net

    if initialize:
        classes = populate_class_labels()
        net = cv2.dnn.readNet(weights_file_abs_path, config_file_abs_path)
        initialize = False

    # enables opencv dnn module to use CUDA on Nvidia card instead of cpu
    if enable_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            max_conf = scores[class_id]
            if max_conf > confidence:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - (w / 2)
                y = center_y - (h / 2)
                class_ids.append(class_id)
                confidences.append(float(max_conf))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms_thresh)

    bbox = []
    label = []
    conf = []

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        if(str(classes[class_ids[i]])==inputlabel):
            bbox.append([int(x), int(y), int(x+w), int(y+h)])
            label.append(str(classes[class_ids[i]]))
            conf.append(confidences[i])
        
    return bbox, label, conf




def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers






















def alone_list(request,label_list,context,count,old_time,inputfile, outputfile):
    logg = open("logfile.txt","a+")
    hist =open(outputfile+"history.txt","w+")
    hist.close()
    hist = open(outputfile+"history.txt","a+")
    for k in label_list:
        hist.write(k)
        hist.write('\n')
        print(k)
    hist.close()
    webcam = cv2.VideoCapture(inputfile)
    webcam2 = cv2.VideoCapture(inputfile)
    frame_width = int(webcam.get(3)) 
    frame_height = int(webcam.get(4)) 
    
    size = (frame_width, frame_height)
    
    only_dict={}
    for labels in label_list:
        output_filename = outputfile+f'{labels}_alone.webm'
        only_dict[labels] = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc(*'vp80'), 10, size)



    if not webcam.isOpened():
        webcam.release()
        webcam2.release()
        result.release()
        for key,value in only_dict :
            only_dict[key].release()

        cv2.destroyAllWindows()
        print("Could not open webcam")
        logg.write("Could not open webcam")
        logg.close()
        return render(request,"object_list.html")

    
    
    i=0
    while webcam.isOpened():
        logg.write("Writing video on alone objects \n")
        print("Video On for lone objects")
# read frame from webcam 
        status, frame = webcam.read()
        stat1, frame2 = webcam2.read()
        if not status:
            runtime = datetime.datetime.now()- old_time
            logg.write("Video Ended...\n")
            print("Video Ended...")
            print("Time taken = ", runtime)
            logg.write("Time taken = "+str(runtime))
            logg.close()
            webcam.release()
            webcam2.release()
            for key in label_list :
                only_dict[key].release()
            cv2.destroyAllWindows()
            return render(request,"object_list.html",{'context':context,'count':count,'outputfile':outputfile})
    
     #   bbox, label, conf = cv.detect_common_objects(frame,0.8,0.5)
       # print(bbox, label, conf)
        
        
        
        
        for name in only_dict:
            bbox1, label1, conf1 = detect_common_object(frame,name)
            if name in label1:
                print(bbox1, label1, conf1)
                out = draw_bbox(frame, bbox1,label1 , conf1)
                print("Wrinting in ",name)
                logg.write("Wrinting boxes for "+name+"\n")
                only_dict[name].write(out)
                out=frame
            else :
                print("Writing frame2 in ",name)
                logg.write("\nWrinting default frame for "+name)
                only_dict[name].write(frame2)

            
        print("End of frame ",i)
        i=i+1

    logg.close()
    webcam.release()
    webcam2.release()
    for key in label_list :
        only_dict[key].release()
    cv2.destroyAllWindows()
    return render(request,"object_list.html",{'context':context,'count':count})
    




def Run(request):

    logg = open("logfile.txt","a+")
    f = open("inputfiles.txt","r")
    rread = f.readlines()
    inputfile= f.read()
    for x in rread:
        inputfile=x
    f.close()
    tempor = open("inputfiles.txt","a+")
    tempor.write("\n")
    tempor.close()
    webcam = cv2.VideoCapture(inputfile)
    printed=[]
    old_time = datetime.datetime.now()
    file1= open("name.txt","a+")
    file1.write(inputfile)
    file1.close()

    
    if not webcam.isOpened():
        webcam.release()
        # result.release()
        for key,value in out_dict :
            outdict[key].release()

        cv2.destroyAllWindows()
        print("Could not open webcam")
        logg.write("Could not open Webcam \n")
        logg.close()
        return render(request,"object_list.html")




    frame_width = int(webcam.get(3)) 
    frame_height = int(webcam.get(4)) 
    f = open("outputfiles.txt","r")
    rread = f.readlines()
    outputfile= f.read()
    for x in rread:
        outputfile=x
    f.close()
    tempor = open("outputfiles.txt","a+")
    tempor.write("\n")
    tempor.close()    
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(outputfile+'output_list.webm',  
                            cv2.VideoWriter_fourcc(*'vp80'), 
                            30, size) 

    file1= open(outputfile+"name.txt","a+")
    file1.write(inputfile)
    file1.close()

    out_dict={}
    
    answer =[]
    count1=0
# loop through frames
    i =0
    while webcam.isOpened():
        logg.write("\nVideo On for frame "+str(i)+"\n")
        print("Video On for frame ",i)
        i=i+1
# read frame from webcam 
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
            logg.write("Video Ended...\n")
            webcam.release()
            result.release()
            for key in out_dict :
                out_dict[key].release()
            logg.close()
            cv2.destroyAllWindows()
            return alone_list(request,answer,context,count1,old_time,inputfile,outputfile)

# apply object detection
        bbox, label, conf = cv.detect_common_objects(frame)
        for k in label:
            logg.write("\n found"+str(k))
            count1=count1+1
            if not k in answer:
                answer.append(k)
        print(bbox, label, conf)


        

# draw bounding box over detected objects
        out = draw_bbox(frame, bbox, label, conf)
        for labels in label:
            if not labels in printed:
                cv2.imwrite(outputfile+f'{labels}_first.jpg',out)
                printed.append(labels)
            if labels in out_dict.keys():
                   # print(f'{label}\'s face already exists')
                out_dict[labels].write(out)
            else:
                  #  print(f'created new face: {label}')
                output_filename = outputfile+f'{labels}_clip.webm'
                out_dict[labels] = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc(*'vp80'), 10, size)
                out_dict[labels].write(out)
# display output
        # cv2.imshow("Real-time object detection", out)
        result.write(out)
# press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logg.write("Wrote to quit\n")
            logg.close()
            print(answer)
            break


# release resources
    webcam.release()
    result.release()
    for key in out_dict :
        out_dict[key].release()
    cv2.destroyAllWindows()
    logg.close()        
    return render(request,"object_list.html")