from django.db import models
import os

# Create your models here.

def impath_and_rename(instance, filename):
    import os
    if os.path.exists("C:/Users/dheer/OneDrive/Desktop/Django/fileuse/fileuse/media/input/image/input.jpg"):
        print("Yeah I found it!!")
        os.remove("C:/Users/dheer/OneDrive/Desktop/Django/fileuse/fileuse/media/input/image/input.jpg")

    upload_to = 'input/image'
    # ext = filename.split('.')[-1]
    # get filename
    filename = 'input.jpg'
    # return the whole path to the file
    return os.path.join(upload_to, filename)

def vipath_and_rename(instance, filename):
    import os
    if os.path.exists("C:/Users/dheer/OneDrive/Desktop/Django/fileuse/fileuse/media/input/video/input.mp4"):
        print("Yeah I found it!!")
        os.remove("C:/Users/dheer/OneDrive/Desktop/Django/fileuse/fileuse/media/input/video/input.mp4")

    upload_to = 'input/video'
    # ext = filename.split('.')[-1]
    # get filename
    filename = 'input.mp4'
    # return the whole path to the file
    return os.path.join(upload_to, filename)



class dataup(models.Model):
    image = models.ImageField(upload_to=impath_and_rename)

class videoup(models.Model):
    video = models.FileField(upload_to=vipath_and_rename)
    
class Video_Train(models.Model):
    video = models.FileField(upload_to='train')
    