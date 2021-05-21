# import enum

# class Gender(enum.Enum):
#     male : 1
#     female : 2
#     other : 3


# class Emotion(enum.Enum):
#     Angry : 1
#     Disgusted : 2
#     Fearful : 3
#     Happy : 4
#     Neutral : 5
#     Sad : 6
#     Surprised : 7

import cv2

class Face_card:
    
    def __init__(self, name, ageGroup, gender, emotion, image = None):
        self.__name = name
        self.__ageGroup = ageGroup
        self.__gender = gender
        self.__emotion = emotion
        self.__image = image

    def getName(self):
        return self.__name

    def getGender(self):
        return self.__gender

    def getAgeGroup(self):
        return self.__ageGroup

    def getEmotion(self):
        return self.__emotion

    def getImage(self):
        # while True:
        if self.__image is None:
            # frame = cv2.imread('noinput.jpg')
            # cv2.imshow('no input', frame)
            self.__image = 'noinput.jpg'

        return self.__image
