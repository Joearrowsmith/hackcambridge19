# Generated by Django 2.1.5 on 2019-01-20 10:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mainService', '0002_graphoutput'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='metrics',
            name='disgust',
        ),
    ]
