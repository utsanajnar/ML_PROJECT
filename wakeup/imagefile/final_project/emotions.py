import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from django.shortcuts import render, HttpResponse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
# ap = argparse.ArgumentParser()
# ap.add_argument("--mode",help="train/display")
# mode = ap.parse_args().mode

# plots accuracy and loss curves

# Define data generators

def analyse(request):
    # print("Analyse func working")
    # from imagefile.models import dataup
    # pics = dataup.objects.all();
    # p = pics[len(pics)-1].image
    # print(p.url)

    # train_dir = 'imagefile/final_project/data/train'
    # val_dir = 'imagefile/final_project/data/test'

    # num_train = 28709
    # num_val = 7178
    # batch_size = 64
    # num_epoch = 50

    # train_datagen = ImageDataGenerator(rescale=1./255)
    # val_datagen = ImageDataGenerator(rescale=1./255)

    # train_generator = train_datagen.flow_from_directory(
    #         train_dir,
    #         target_size=(48,48),
    #         batch_size=batch_size,
    #         color_mode="grayscale",
    #         class_mode='categorical')

    # validation_generator = val_datagen.flow_from_directory(
    #         val_dir,
    #         target_size=(48,48),
    #         batch_size=batch_size,
    #         color_mode="grayscale",
    #         class_mode='categorical')

    # Create the model
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

    # # If you want to train the same model or try other models, go for this
    # if mode == "train":
    #     model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    #     model_info = model.fit_generator(
    #             train_generator,
    #             steps_per_epoch=num_train // batch_size,
    #             epochs=num_epoch,
    #             validation_data=validation_generator,
    #             validation_steps=num_val // batch_size)
    #     plot_model_history(model_info)
    #     model.save_weights('imagefile/final_project/model.h5')

    # emotions will be displayed on your face from the webcam feed
    # elif mode == "display":
    model.load_weights('imagefile/final_project/model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    # cap = cv2.VideoCapture('imagefile/final_project/input1.jpg')
    # width = (int)(cap.get(3))
    # height = (int)(cap.get(4))

    # fourcc = cv2.VideoWriter_fourcc(*'vp80')
    # out = cv2.VideoWriter('/static/emotion_output.webm', fourcc, 10.0, (width,height))
    # number_frames = 1

# while True:
    # Find haar cascade to draw bounding box around face
    # ret, frame = cap.read()

    # if not ret:
    #     break

    # if number_frames % 3 != 0:
    #     number_frames += 1
    #     continue
    frame = cv2.imread('/Volumes/TEMPUSB/fileuse/fileuse/media/input/image/input.jpg')
    frame = cv2.resize(frame, (640,480))
    facecasc = cv2.CascadeClassifier('imagefile/final_project/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:

        print(f"shape of faces {faces.shape}")
        print(f"x, y, w, h: {x}, {y}, {w}, {h}")
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # out.write(frame)
    # cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    cv2.imwrite('/Volumes/TEMPUSB/fileuse/fileuse/media/output/image/output.jpg',frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # number_frames += 1

    # cap.release()
    # out.release()
    cv2.destroyAllWindows()
    
    return render(request, "result.html")
