# Generated by Django 3.1.1 on 2020-10-02 18:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imagefile', '0007_auto_20200929_1634'),
    ]

    operations = [
        migrations.CreateModel(
            name='Video_Train',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('video', models.FileField(upload_to='media/train')),
            ],
        ),
    ]
