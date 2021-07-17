# Generated by Django 3.1.3 on 2021-06-02 20:10

import account.models
from django.db import migrations
import imagekit.models.fields


class Migration(migrations.Migration):

    dependencies = [
        ('account', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='customuser',
            name='face_bank',
            field=imagekit.models.fields.ProcessedImageField(blank=True, null=True, upload_to=account.models.facebank_image),
        ),
    ]