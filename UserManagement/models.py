from django.db import models
from django.contrib.auth.models import AbstractUser
from auditlog.registry import auditlog

# Create your models here.
class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    photo = models.ImageField(upload_to='User/Photo', null=True, blank=True)
    is_active = models.BooleanField(default=True)
    USERNAME_FIELD='email'
    REQUIRED_FIELDS=['first_name','username','last_name']
    is_first_time = models.BooleanField(default = True)
    last_reset_password = models.DateTimeField(null=True, blank=True)
    is_dashboard =  models.BooleanField(default = False)
    is_paid =  models.BooleanField(default = False)


auditlog.register(CustomUser)