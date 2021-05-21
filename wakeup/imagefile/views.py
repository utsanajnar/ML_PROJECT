# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from imagefile.final_project.emotions import analyse
from django.shortcuts import render, HttpResponse
from django.core.files.storage import FileSystemStorage
from .models import videoup
from .models import dataup , Video_Train
import os
from datetime import date
from django.conf import settings
# from .forms import UploadFileForm

# from somewhere import handle_uploaded_file

# Create your views here.

# def upload_file(request):
#     if request.method == 'POST':
#         form = UploadFileForm(request.POST, request.FILES)
#         if form.is_valid():
#             handle_uploaded_file(request.FILES['file'])
#             return HttpResponseRedirect('/success/url/')
#     else:
#         form = UploadFileForm()
#     return render(request, 'upload.html', {'form': form})

# def uploadingImage(request):
#     print("Request Handling...........")
#     p = request.FILES['image']
    

#     img = dataup( image = p)
#     img.save()

#     return render(request, 'home.html')  



def ImageTraining(request):
    return render(request, 'image_train.html')

def VideoTraining(request):
    
    if request.method == "POST": 
        print("Running Func for Video Uploading")
        p = request.FILES['videotrain']

        vid = Video_Train( video = p)
        vid.save()

    return render(request, 'video_train.html')

def Trainingpage(request):
    return render(request, 'result.html')

def uploadingImage1(request):
    if request.method == "POST": 
    # if the post request has a file under the input name 'document', then save the file. 
         request_file = request.FILES['image'] if 'image' in request.FILES else None
         if request_file: 
                    # save attatched file 
                   directory = request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")
                   parent_dir = settings.MEDIA_ROOT
                   path = os.path.join(parent_dir,directory).replace("\\", "/" )

                   
 
                   os.mkdir(path)
                    # create a new instance of FileSystemStorage 
                   fs = FileSystemStorage(location= path) 
                   file = fs.save(request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+request_file.name[-4:], request_file) 
                    # the fileurl variable now contains the url to the file. This can be used to serve the file when needed. 
                   fileurl = fs.url(file)
                   tempor = open("inputfiles.txt","a+")
                   tempor.write('media/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+'/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+request_file.name[-4:])
                   tempor.close()
                   context = {'filepath':file[:-4]}
                   tempor = open("outputfiles.txt","a+")
                   tempor.write('media/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+'/')
                   tempor.close()
    return render(request, 'home.html')

# def uploadingVideo(request):
    # print("Running Func for Video Uploading")
    # p = request.FILES['video']

    # vid = dataup( video = p)
    # vid.save()

    # return render(request, 'video.html')      

def imagefile(request):
    return render(request, 'home.html')    

def uploadingVideo1(request):
    
    if request.method == "POST":
        request_file = request.FILES['file'] if 'file' in request.FILES else None
        if request_file: 
                # save attatched file 
                directory = request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")
                parent_dir = settings.MEDIA_ROOT
                path = os.path.join(parent_dir,directory).replace("\\", "/" )

                

                os.mkdir(path)
                # create a new instance of FileSystemStorage 
                fs = FileSystemStorage(location= path) 
                file = fs.save(request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+request_file.name[-4:], request_file) 
                # the fileurl variable now contains the url to the file. This can be used to serve the file when needed. 
                fileurl = fs.url(file)
                tempor = open("inputfiles.txt","a+")
                tempor.write('media/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+'/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+request_file.name[-4:])
                tempor.close()
                context = {'filepath':file[:-4]}
                tempor = open("outputfiles.txt","a+")
                tempor.write('media/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+'/')
                tempor.close()

    return render(request,'video.html')

def view(request):
    return render(request, 'view.html') 

def result(request):
    return render(request, 'result.html') 
# def analyse(request):
#     analyseImage('imagefile/final_project/input_image_3.jpg', 'output_image.jpg', 0.43)
#     return render(request, 'result.html')     
