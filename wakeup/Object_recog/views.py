from django.shortcuts import render, HttpResponse
from Object_recog.models import videoup1
from django.core.files.storage import FileSystemStorage
from datetime import date
from django.conf import settings
import os

# Create your views here.
def display(request):
    context = {"name":"rushi", "Place":"Wake-up"}
    return render(request, "Webview.html",context)
    #return HttpResponse("Aa gaya yaha?\n\n\n\n ho gaye khush?")

def Image(request):
     context = {'name':'Rushi', 'filepath':'Wake-up'}
     return render(request,"Image_detection.html",context)

def Video(request):
     context = {'name':'Rushi', 'Place':'Wake-up'}
     return render(request,"Video_detection.html",context)

def object_List(request):
     context = {'name':'Rushi', 'Place':'Wake-up'}
     return render(request,"object_list.html",context)

def object_video(request):
     context = {'name':'Rushi', 'Place':'Wake-up'}
     return render(request,"object_video.html",context)

def rr(request,inputlabel):
     
     return render(request,"object_list.html",{'context':inputlabel})

def dd(request,inputlabel):

     return render(request,"object_list",{'context':inputlabel})



def historyExist(request, outputfile):
     print("Inside history exisit")
     
     
     walke= open(outputfile+"history.txt","r")
     #op=walke.read()
     k = walke.readlines()
     i=-1
     answer={}
     print("\n\n Trying before k")
     for item in k:
          
          print(item)
          answer[i]=item
          i=i+1
     walke.close()
     print("\n\n outputfile = "+outputfile)
     return render(request,"image_list.html",{"context":answer, "count": i+1, "outputfile":outputfile})

def historyExist2(request, outputfile):
     print("iNSIDE HISTORY EXIXSTT2")
     walke= open(outputfile+"history.txt","r")
     #op=walke.read()
     k = walke.readlines()
     i=-1
     answer={}
    # print("Trying before k in video")
     for item in k:
          
          print(item)
          answer[i]=item
          i=i+1
     walke.close()
     return render(request,"object_list.html",{"context":answer, "count": i+1, "outputfile":outputfile})








def uploadingImage(request):
    if request.method == "POST": 
    # if the post request has a file under the input name 'document', then save the file. 
         request_file = request.FILES['image'] if 'image' in request.FILES else None
         print("Inside upload Image")
         if request_file: 
                    # save attatched file 
                   directory = request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")
                   parent_dir = settings.MEDIA_ROOT
                   path = os.path.join(parent_dir,directory).replace("\\", "/" )
                   valid = open("inputfiles.txt","r")
                   rread = valid.readlines()
                   for x in rread:
                        print(x)
                        print("\n\n")
                        print('media/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+'/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+request_file.name[-4:])
                        if(x == ('media/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+'/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+request_file.name[-4:]+'\n')):
                             outputfile = 'media/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+'/'
                             return historyExist(request,outputfile)

                   
 
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
    return render(request, 'Image_detection.html',context)

def uploadingVideo(request):
    
    if request.method == "POST": 
         print("Inside upload video again")
    # if the post request has a file under the input name 'document', then save the file. 
         request_file = request.FILES['video'] if 'video' in request.FILES else None
         print("request file = "+str(request_file))
         if request_file: 
                    # save attatched file 
                   directory = request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")
                   parent_dir = settings.MEDIA_ROOT
                   path = os.path.join(parent_dir,directory).replace("\\", "/" )

                   valid = open("inputfiles.txt","r")
                   rread = valid.readlines()
                   for x in rread:
                        print(x)
                        print("\n\n")
                        print('media/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+'/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+request_file.name[-4:])
                        if(x == ('media/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+'/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+request_file.name[-4:]+'\n')):
                             outputfile = 'media/'+request_file.name[:-4]+date.today().strftime("_%d_%m_%Y")+'/'
                             return historyExist2(request,outputfile)

                   
 
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

    return render(request, 'Video_detection.html') 


def history_image(request):
    f = open("inputfiles.txt", "r")
    rread = f.readlines()
    context = {}
    file_type={}
    extension={}
    distinct_dates = []
    i = 0
    for x in rread:
        if not x == '\n':
            
            y = x.split('/')[-1].split('.')[0]
            typ = x[-4:-1]
            
            
            file_name=y
            
            file_type[file_name]=typ
            z = y.split('_')
            date = z[-3]+'_'+z[-2]+'_'+z[-1]
            extension[date]="media/"+file_name
            #extension[file_name] = x.split('/')[-1].split('.')[1]
            if date in distinct_dates:
                context[date].append(file_name)
            else:
                context[date] = []
                context[date].append(file_name)
                distinct_dates.append(date)

        i = i+1
    return render(request, 'history_image.html', {"context": context, "file_type": file_type,"extension":extension })

def history_video(request):
    f = open("inputfiles.txt", "r")
    rread = f.readlines()
    context = {}
    distinct_dates = []
    i = 0
    for x in rread:
        if not x == '\n':
            y = x.split('/')[-1].split('.')[0]
            file_name=y
            z = y.split('_')
            date = z[-3]+'_'+z[-2]+'_'+z[-1]
            if date in distinct_dates:
                context[date].append(file_name)
            else:
                context[date] = []
                context[date].append(file_name)
                distinct_dates.append(date)
        i = i+1
    return render(request, 'history_video.html', {"context": context})


def histo(request,opfl):
     x=open('media/'+opfl+'/name.txt','r')
     name=x.readline()
     x.close()
     if name.split('.')[-1]=='jpg':
          return historyExist(request,'media/'+opfl+'/')
     return historyExist2(request,'media/'+opfl+'/')