from django.db import models
from website.common.model_manager import ModelManager

# Create your models here.
# class Malicious(models.Model):
#     task = models.CharField(max_length = 500)

#     def __str__(self):
#     	return self.task

class Malicious(models.Model):
    task = models.CharField(primary_key=True, null=False, max_length=500)
    objects = ModelManager()

    class Meta:
        pass