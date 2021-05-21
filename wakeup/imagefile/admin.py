from django.contrib import admin

# Register your models here.

from imagefile.models import dataup
from imagefile.models import videoup , Video_Train

admin.site.register(dataup)
admin.site.register(videoup)
admin.site.register(Video_Train)