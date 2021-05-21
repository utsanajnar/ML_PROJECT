"""Object_Recognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

from Object_recog import display_list
from Object_recog.cvlib.examples import save
from Object_recog.cvlib.examples import Clips
from Object_recog import save_run1
from django.conf import settings
from django.conf.urls.static import static
from imagefile.final_project import emotions
from imagefile.final_project import test3
from imagefile.final_project import test4

from imagefile.views import(
    imagefile,
    uploadingVideo1,
    uploadingImage1,

)
from Object_recog.views import(
    Image,
    display,
    Video,
    object_video,
    uploadingImage,
    uploadingVideo,
    dd,
    rr,
    histo,
    
    history_image,
    history_video


)
from employee.views import(
    emp,
    show,
    edit,
    update,
    destroy,
    videoScreen,
    videoScreen5,
    videoScreen9
)
    

urlpatterns = [
    path('', display),
    path('admin/', admin.site.urls),
    path('Object_recog/', display),
    path('Image', Image),
    path('Video', Video),
    path('object_video', object_video),
    path('Object_List',display_list.Run),
    path('image_list',save_run1.Run),
    path('Video_Save',save.Run),
    path('Clips/<str:inputlabel>',dd, name='Clips'),
    path('Object/<str:inputlabel>',rr, name='Object'),
    path('upload' , uploadingImage, name="UploadImage"),
    path('upload_video' , uploadingVideo, name="UploadVideo"),


    path('/<str:opfl>',histo, name= 'Rushi'),
   # path('/<str:abcd>',histo2, name= 'walke'),

    path('faceimage', imagefile,name="Home"),
    path('analysis' ,test3.analyse_face),
    path('video/', uploadingVideo1, name="Video"),
    path('video/analysevideo/', test4.analyse_video, name="VideoAnalysis"),
    path('upload1' , uploadingImage1, name="UploadImage"),
    # path('imageTrain' , views.ImageTraining, name="TrainImage"),
    # path('videoTrain' , views.VideoTraining, name="TrainVideo"),
    # # path('trainvideodata', train.run, name="StartTraining"),
    # path('videouploadtraining' , views.VideoTraining, name="TrainVideo"),
    # path('training' , views.Trainingpage, name="TrainingPage"),
    # # path('uploadvideo' , views.uploadingVideo, name="UploadVideo"),

    path('history_image', history_image, name='history_image'),
    path('history_video', history_video, name='history_video'),


    path('emp', emp),  
    path('show',show),  
    path('edit/<int:id>', edit),  
    path('update/<int:id>', update),  
    path('delete/<int:id>', destroy), 
    path('videoScreen',videoScreen , name="videoScreen"),
    path('vid5',videoScreen5 , name="videoScreen5"),
    path('vid9',videoScreen9 , name="videoScreen9")

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
