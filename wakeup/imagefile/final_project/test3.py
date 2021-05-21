# -*- coding: utf-8 -*-
from imagefile.final_project.face_card import Face_card
import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
import tensorflow as tf
import pickle
import onnx
import onnxruntime as ort
import datetime as dt
from onnx_tf.backend import prepare
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from django.shortcuts import render, HttpResponse

def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

# plots accuracy and loss curves

def predict_emotions(gray_image, emotion_model):
    """
    Predicts the emotion
    Input:
        gray_image: a gray image containing a face
        emtion_model: model which will predict the motion

    Output:
        1 of the 7 emotions out of - Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised
    """
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    expanded_img = np.expand_dims(np.expand_dims(cv2.resize(gray_image, (48, 48)), -1), 0)

    # global emotionDetectionModel
    emotion_model.load_weights('imagefile/final_project/model.h5')
    prediction_index = emotion_model.predict(expanded_img)
    prediction_index = int(np.argmax(prediction_index))

    return emotion_dict[prediction_index]

def getEmotionDectionModel():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    # model.load_weights('model.h5')

    return model

def getAgeGenderModel():
    ageProto="imagefile/final_project/age_deploy.prototxt"
    ageModel="imagefile/final_project/age_net.caffemodel"

    genderProto="imagefile/final_project/gender_deploy.prototxt"
    genderModel="imagefile/final_project/gender_net.caffemodel"

    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    return genderNet, ageNet

def detectAgeGender(face, genderModel, ageModel):

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    genderList=['Male','Female']
    ageList=['(0-3)', '(4-7)', '(8-14)', '(15-24)', '(25-37)', '(38-47)', '(48-59)', '(60-100)']

    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

    # global genderModel
    genderModel.setInput(blob)
    genderPreds=genderModel.forward()
    predicted_gender=genderList[genderPreds[0].argmax()]

    # global ageModel
    ageModel.setInput(blob)
    agePreds=ageModel.forward()
    predicted_age=ageList[agePreds[0].argmax()]

    return predicted_gender, predicted_age 

def initiateVideoCapture(inputFilePath):
    videoCaptureObj = cv2.VideoCapture(inputFilePath)
    frameWidth = (int)(videoCaptureObj.get(3))
    frameHeight = (int)(videoCaptureObj.get(4))

    return videoCaptureObj, frameWidth, frameHeight

def displayFaceCards(faceCards):
    print("")
    print('displaying face cards')  
    print("")
    
    for face_cards in faceCards:
        print(face_cards.getName())
        print(face_cards.getAgeGroup())
        print(face_cards.getGender())
        print(face_cards.getEmotion())
        print(face_cards.getImage())
        print("")

    return

def releaseHandles(outPutVideoHandle, out_dict):
    outPutVideoHandle.release()

    for out_vid in out_dict:
        out_dict[out_vid].release()
    
    cv2.destroyAllWindows()
    return

def loadEmbeddings():
    with open("imagefile/final_project/embeddings/embeddings.pkl", "rb") as f:
        (saved_embeds, names, distinct_names) = pickle.load(f)
    return saved_embeds, names, distinct_names

def loadFaceRecognitionModel():
    # preparing the pretrained model
    onnx_path = 'imagefile/final_project/models/ultra_light/ultra_light_models/ultra_light_640.onnx'
    onnx_model = onnx.load(onnx_path)
    predictor = prepare(onnx_model)
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    # we have used dlib for shape predictor
    shape_predictor = dlib.shape_predictor('imagefile/final_project/models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
    fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

    return ort_session, input_name, fa

# def analyseVideo(inputVideoPath, outputVideoFileName, outputVideoFormat, threshold):

#     OUTPUT_FILE_BASE = 'imagefile/final_project/output/Video'
#     os.path.join(OUTPUT_FILE_BASE, outputVideoFileName)
#     ort_session, input_name, fa =loadFaceRecognitionModel()
#     saved_embeds, names, distinct_names = loadEmbeddings()

#     # Output video Dicitonary
#     out_dict = {}

#     # video spcification
#     # inputVideoPath = 'input.MOV'
#     # output_video = 'output.mp4'
#     fourcc = cv2.VideoWriter_fourcc(*outputVideoFormat)
#     video_capture, width, height = initiateVideoCapture(inputVideoPath)
#     out = cv2.VideoWriter(os.path.join(OUTPUT_FILE_BASE, outputVideoFileName),fourcc, 20.0, (width,height))
    

#     # Face card array
#     faceCards = []



#     graph = tf.get_default_graph()
#     with graph.as_default():
#         with tf.Session() as sess:

#             saver = tf.train.import_meta_graph('imagefile/final_project/models/mfn/m1/mfn.ckpt.meta')
#             saver.restore(sess, 'imagefile/final_project/models/mfn/m1/mfn.ckpt')

#             images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#             embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#             phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#             embedding_size = embeddings.get_shape()[1]

#             #loading video capture
#             # video_capture = cv2.VideoCapture('input1.MOV')
#             # width = (int)(video_capture.get(3))
#             # height = (int)(video_capture.get(4))  

            

#             number_of_frames = 1.0
#             start_time = dt.datetime.now()
#             frames_per_sec = 0.0

#             # predicted_face = ""
#             # predicted_emotion = ""
            
#             while video_capture.isOpened():

#                 # fps = video_capture.get(cv2.CAP_PROP_FPS)
#                 # print(f"current fps {fps}")
#                 ret, frame = video_capture.read()

#                 if ret == False:
#                     break

#                 # preprocess faces
#                 h, w, _ = frame.shape
#                 img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 img = cv2.resize(img, (640, 480))
#                 img_mean = np.array([127, 127, 127])
#                 img = (img - img_mean) / 128
#                 img = np.transpose(img, [2, 0, 1])
#                 img = np.expand_dims(img, axis=0)
#                 img = img.astype(np.float32)

#                 # detect faces
#                 confidences, boxes = ort_session.run(None, {input_name: img})
#                 boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

                
#                 boxes[boxes<0] = 0
#                 # print(f'shape of boxes {boxes.shape}')
#                 # print(f'shape of boxes 0{boxes.shape[0]}')

#                 # draw
#                 for i in range(boxes.shape[0]):
                    

#                     faces = []

#                     box = boxes[i, :]
#                     x1, y1, x2, y2 = box
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
                    
#                     # pre procssing for facial recognition
#                     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                     aligned_face = fa.align(frame, gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
#                     aligned_face = cv2.resize(aligned_face, (112,112))

#                     aligned_face = aligned_face - 127.5
#                     aligned_face = aligned_face * 0.0078125

#                     faces.append(aligned_face)

#                     # FACE RECOGNITION
#                     faces = np.array(faces)
#                     # print(f"type of faces in face embedding {type(faces)}")
                
#                     feed_dict = { images_placeholder: faces, phase_train_placeholder:False }
#                     embeds = sess.run(embeddings, feed_dict=feed_dict)

#                     for embedding in embeds:
#                         predicted_face  = "" 
#                         diff = np.subtract(saved_embeds, embedding)
#                         dist = np.sum(np.square(diff), 1)
#                         idx = np.argmin(dist)
#                         if dist[idx] < threshold:
#                             predicted_face = names[idx]
#                             # predictions.append(names[idx])
#                         else:
#                             predicted_face = "unknown"
#                             # predictions.append("unknown")

#                     # EMOTION RECOGNITION
#                     roi_gray = gray[y1:y2, x1:x2]
#                     predicted_emotion = predict_emotions(roi_gray, emtionDetectionModel)
                    
#                     # AGE & GENDER RECOGNITION

#                     face = frame[y1:y2, x1:x2]
#                     predicted_gender, predicted_age = detectAgeGender(face)


                    
#                     output_text = f"{predicted_face} is {predicted_emotion} {predicted_gender} b/w {predicted_age}"

#                     # Draw a label with a name below the face
#                     # cv2.rectangle(frame, (x1, y2 - 40), (x2, y2), (80,18,236), cv2.FILLED)
#                     font = cv2.FONT_HERSHEY_DUPLEX
#                     cv2.putText(frame, output_text, (x1 + 6, y2 + 15), font, 0.5, (255, 255, 255), 1)
                    
#                     # cv2.putText(frame, emotion_text, (x1 + 6, y2 + 30), font, 0.5, (255, 255, 255), 1)
#                     # cv2.putText(frame, text1, (x1,y1), font, 0.3, (255, 255, 255), 1)
#                     # cv2.putText(frame, text2, (x2,y2), font, 0.3, (255, 255, 255), 1)
                    
#                     if predicted_face in out_dict.keys():
#                         out_dict[predicted_face].write(frame)
#                     else:
#                         output_filename = f'{predicted_face}.mp4'
#                         out_dict[predicted_face] = cv2.VideoWriter(os.path.join(OUTPUT_FILE_BASE, output_filename),fourcc, 20.0, (width,height))
#                         out_dict[predicted_face].write(frame)
#                         img_output_file = f'{predicted_face}.jpg'
#                         cv2.imwrite(os.path.join(OUTPUT_FILE_BASE, img_output_file), face)
#                         faceCards.append(Face_card(predicted_face, predicted_age, predicted_gender, predicted_emotion, img_output_file))

#                 out.write(frame)
                
#                 current_time = dt.datetime.now()
#                 time_elapsed = current_time - start_time
#                 time_elapsed_in_float = float(time_elapsed.seconds)
#                 frames_per_sec = number_of_frames / time_elapsed_in_float
#                 if number_of_frames == 1:
#                     max_fps = frames_per_sec
#                 print(f"Frame #      :{number_of_frames}")
#                 print(f"Time elapsed :{time_elapsed}")
#                 if max_fps < frames_per_sec:
#                     max_fps = frames_per_sec
#                 print(f"Current fps  :{frames_per_sec}")
#                 print(f"Max fps      :{max_fps}")
#                 print("")
#                 number_of_frames = number_of_frames + 1
#                 # Hit 'q' on the keyboard to quit!
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#     # Release handle to the webcam & out
#     print('video analysis done!')

#     releaseHandles(out, out_dict)
#     displayFaceCards(faceCards)

#     return

def analyseImage(inputImagePath, outputImagePath, threshold):
    cv2.ocl.setUseOpenCL(False)
    emotionDetectionModel = getEmotionDectionModel()
    genderModel, ageModel = getAgeGenderModel()
    
    INPUT_IMAGE_FILE_BASE = os.getcwd()+'/'
    # OUTPUT_IMAGE_FILE_BASE = 'C:/Users/dheer/OneDrive/Desktop/Django/fileuse/fileuse/media/output/image/'

    ort_session, input_name, fa =loadFaceRecognitionModel()
    saved_embeds, names, distinct_names = loadEmbeddings()
    inputImagePath = INPUT_IMAGE_FILE_BASE+inputImagePath
    print(inputImagePath)
    print(outputImagePath)
    frame = cv2.imread(inputImagePath)
    # frame = cv2.imread(inputImagePath)
    unknown_count = 1
    # Face card array
    faceCards = []


    graph = tf.get_default_graph()
    with graph.as_default():
        with tf.Session() as sess:

            saver = tf.train.import_meta_graph('imagefile/final_project/models/mfn/m1/mfn.ckpt.meta')
            saver.restore(sess, 'imagefile/final_project/models/mfn/m1/mfn.ckpt')

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # preprocess faces
            h, w, _ = frame.shape
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img_mean = np.array([127, 127, 127])
            img = (img - img_mean) / 128
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)

            # detect faces
            confidences, boxes = ort_session.run(None, {input_name: img})
            boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

            
            boxes[boxes<0] = 0
            # print(f'shape of boxes {boxes.shape}')
            # print(f'shape of boxes 0{boxes.shape[0]}')

            # draw
            for i in range(boxes.shape[0]):
                faces = []

                box = boxes[i, :]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
                
                # pre procssing for facial recognition
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                aligned_face = fa.align(frame, gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
                aligned_face = cv2.resize(aligned_face, (112,112))

                aligned_face = aligned_face - 127.5
                aligned_face = aligned_face * 0.0078125

                faces.append(aligned_face)

                # FACE RECOGNITION
                faces = np.array(faces)
                # print(f"type of faces in face embedding {type(faces)}")
            
                feed_dict = { images_placeholder: faces, phase_train_placeholder:False }
                embeds = sess.run(embeddings, feed_dict=feed_dict)

                for embedding in embeds:
                    predicted_face  = "" 
                    diff = np.subtract(saved_embeds, embedding)
                    dist = np.sum(np.square(diff), 1)
                    idx = np.argmin(dist)
                    if dist[idx] < threshold:
                        predicted_face = names[idx]
                    else:
                        predicted_face = "unknown"

                # EMOTION RECOGNITION
                roi_gray = gray[y1:y2, x1:x2]
                predicted_emotion = predict_emotions(roi_gray, emotionDetectionModel)
                
                # AGE & GENDER RECOGNITION

                face = frame[y1:y2, x1:x2]
                predicted_gender, predicted_age = detectAgeGender(face, genderModel, ageModel)


                
                output_text = f"{predicted_face} is {predicted_emotion} {predicted_gender} b/w {predicted_age}"

                # Draw a label with a name below the face
                # cv2.rectangle(frame, (x1, y2 - 40), (x2, y2), (80,18,236), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(frame, output_text, (x1 + 6, y2 + 20), font, 1, (255, 255, 255), 1)
                if predicted_face == 'unknown':
                    predicted_face = f'{predicted_face}_{unknown_count}'
                    unknown_count += 1

                output_file_path = f'{predicted_face}.jpg'
                faceCards.append(Face_card(predicted_face, predicted_age, predicted_gender, predicted_emotion, output_file_path))
                print(f'here we write this image {INPUT_IMAGE_FILE_BASE+outputImagePath+output_file_path}')
                cv2.imwrite(INPUT_IMAGE_FILE_BASE+outputImagePath+output_file_path, face)

            cv2.imwrite(INPUT_IMAGE_FILE_BASE+outputImagePath+'output.jpg', frame)

    # Release handle to the webcam & out

    print('Image analysis done!')
    cv2.destroyAllWindows()
    # displayFaceCards(faceCards)   
    

    return faceCards
# Create  models
# emotionDetectionModel = getEmotionDectionModel()
# genderModel, ageModel = getAgeGenderModel()
# print('models loaded')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)



# print('Analysing video')
# analyseVideo('input1.MOV', 'output.mp4', 'mp4v', 0.43)

def analyse_face(request):
    print('in Analyse face function')
    context = {}
    f = open("inputfiles.txt","r")
    rread = f.readlines()
    inputfile= f.read()
    for x in rread:
        inputfile=x
    f.close()
    tempor = open("inputfiles.txt","a+")
    tempor.write("\n")
    tempor.close()

    f = open("outputfiles.txt","r")
    rread = f.readlines()
    outputfile= f.read()
    for x in rread:
        outputfile=x
    f.close()
    tempor = open("outputfiles.txt","a+")
    tempor.write("\n")
    tempor.close()

    print(inputfile)
    print(outputfile)
    
    faceCards = analyseImage(inputfile, outputfile, 0.43)
    print(f'here is output file {outputfile}')
    # outputfile = os.getcwd()+'/'+outputfile
    print(outputfile)
    for face_cards in faceCards:
        context[face_cards.getName()]= {}
        context[face_cards.getName()]['name'] = face_cards.getName()
        context[face_cards.getName()]['agegroup'] = face_cards.getAgeGroup()
        context[face_cards.getName()]['gender'] = face_cards.getGender()
        context[face_cards.getName()]['emotion'] = face_cards.getEmotion()

    print("Im Doone")
    return render(request, "home.html", {'output': outputfile,'context': context} )


