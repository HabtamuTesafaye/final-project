# Generated by Django 4.2.6 on 2024-08-24 10:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('UserManagement', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='customuser',
            name='is_paid',
            field=models.BooleanField(default=False),
        ),
    ]
