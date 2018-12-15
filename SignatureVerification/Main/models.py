from django.db import models

# Create your models here.


def user_directory(instance, filename):
    return str(instance.c_id) + "/" + filename


#Path for Registration Images
def img_directory(instance, filename):
    return str(instance.customerdetails.c_id) + "/" + filename


class CustomerDetails(models.Model):
    c_id = models.CharField(max_length=250)
    c_name = models.CharField(max_length=250)
    #image = models.FileField(upload_to=user_directory)


class VerificationDetails(models.Model):
    c_id = models.CharField(max_length=250)
    image = models.FileField(upload_to=user_directory)


#Model for Images
class Attachment(models.Model):
    customerdetails = models.ForeignKey(CustomerDetails,on_delete=models.CASCADE)
    file = models.FileField(upload_to=img_directory)
