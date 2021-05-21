from django.db import models
import os


# Create your models here.

def path_and_rename(instance, filename):
    import os
    if os.path.exists(os.getcwd()+"/media/image/input.jpg"):
        print("Yeah I found it!!")
        os.remove(os.getcwd()+"/media/image/input.jpg")

    upload_to = 'image'
    # ext = filename.split('.')[-1]
    # get filename
    filename = 'input.jpg'
    # return the whole path to the file
    return os.path.join(upload_to, filename)

def vi_path_and_rename(instance, filename):
    import os
    if os.path.exists(os.getcwd()+"/media/video/input.webm"):
        print("Yeah I found it!!")
        os.remove(os.getcwd()+"/media/video/input.webm")

    upload_to = 'video'
    # ext = filename.split('.')[-1]
    # get filename
    filename = 'input.webm'
    # return the whole path to the file
    return os.path.join(upload_to, filename)


class dataup1(models.Model):
    image = models.ImageField(upload_to=path_and_rename)

class videoup1(models.Model):
    video = models.FileField(upload_to=vi_path_and_rename)