from django.db import models
import time


# Create your models here.


def user_directory(instance, filename):
	timestr = time.strftime("%Y%m%d-%H%M%S")
	return str(instance.c_id) + "/Verification/" + str(timestr) + filename 


#Path for Registration Images
def img_directory(instance, filename):
	timestr = time.strftime("%Y%m%d-%H%M%S")
	return str(instance.customerdetails.c_id) + "/Registration/" + str(timestr) + filename



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
