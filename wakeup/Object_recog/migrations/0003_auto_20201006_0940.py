# Generated by Django 2.2.2 on 2020-10-06 09:40

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Object_recog', '0002_videoup'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='dataup',
            new_name='dataup1',
        ),
        migrations.RenameModel(
            old_name='videoup',
            new_name='videoup1',
        ),
    ]