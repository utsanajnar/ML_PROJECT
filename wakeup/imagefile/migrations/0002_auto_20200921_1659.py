# Generated by Django 3.1.1 on 2020-09-21 11:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imagefile', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dataup',
            name='firstname',
            field=models.CharField(default='', max_length=20),
        ),
        migrations.AlterField(
            model_name='dataup',
            name='image',
            field=models.ImageField(default='', upload_to='upimg'),
        ),
        migrations.AlterField(
            model_name='dataup',
            name='lastname',
            field=models.CharField(default='', max_length=20),
        ),
    ]