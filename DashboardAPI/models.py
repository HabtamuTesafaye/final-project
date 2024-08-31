from django.db import models

# Create your models here.
class WebsiteInfo(models.Model):
    key = models.CharField(max_length=100, unique=True)
    value = models.TextField()

    def __str__(self):
        return self.key