# Generated by Django 3.1.1 on 2020-09-28 19:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imagefile', '0004_auto_20200929_0044'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dataup',
            name='image',
            field=models.ImageField(upload_to='static/image'),
        ),
    ]