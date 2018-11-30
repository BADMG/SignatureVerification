from django.db import models

# Create your models here.

class CustomerDetails(models.Model):
    c_id = models.CharField(max_length=250)
    c_name = models.CharField(max_length=250)

