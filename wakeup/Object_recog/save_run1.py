# import necessary packages
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
from django.shortcuts import render, HttpResponse
from cvlib.utils import download_file
import numpy as np
import os

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
   # print(inputfile)
    frame = cv2.imread(inputfile)
 
    answer={}
    bbox, label, conf = cv.detect_common_objects(frame)
  #  print(bbox, label, conf)
    out = draw_bbox(frame, bbox, label, conf)
    #print(out)
    #cv2.imshow("Real-time object detection", out)
    
    print("I am after imshow")
    f = open("outputfiles.txt","r")
    rread = f.readlines()
    outputfile= f.read()
    for x in rread:
        outputfile=x
    f.close()

    tempor = open("outputfiles.txt","a+")
    tempor.write("\n")
    tempor.close()

    file1= open(outputfile+"name.txt","a+")
    file1.write(inputfile)
    file1.close()

  #  print("Before imwrite")
  #  print("outputfile = "+ outputfile)
    cv2.imwrite(outputfile+'image_output.jpg', out)
    i=0
    hist =open(outputfile+"history.txt","w+")
    hist.close()
    hist = open(outputfile+"history.txt","a+")
    #hist.write(outputfile+'\n')
    for item in label:
        hist.write(item)
        hist.write('\n')
        logg.write(item+" detected \n")
        frame2=cv2.imread(inputfile)
        bbox1, label1, conf1 = detect_common_object(frame2,item)
        screenshot = draw_bbox(frame2,bbox1,label1,conf1)
        cv2.imwrite(outputfile+f'{item}_image.jpg',screenshot)
    #    print(item)
        
        answer[i]=item
        i=i+1
    logg.close()
    hist.close()
    cv2.destroyAllWindows()
	#return
	
    return render(request,"image_list.html",{"context":answer, "count": i, "outputfile":outputfile})
	