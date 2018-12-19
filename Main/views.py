from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.generic import TemplateView
from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from Main.forms import CustomerForm, VerificationForm, NumberOfForms
from Main.models import CustomerDetails, VerificationDetails, Attachment
import numpy as np
from keras import backend as K
import keras.models as models
import tensorflow as tf
import keras.layers as layers
import cv2

class MainView(TemplateView):
    template_name = 'Main/index.html'
    img_width = 300
    img_height = 150
    batch = 128

    
    def modelCreator(self):
        modelA = models.Sequential()
        modelA.add(layers.Conv2D(16, (3, 3), input_shape=(self.img_width, self.img_height, 1)))
        modelA.add(layers.BatchNormalization())
        modelA.add(layers.Activation("relu"))
        modelA.add(layers.MaxPooling2D((2, 2)))
        modelA.add(layers.Conv2D(32, (3, 3)))
        modelA.add(layers.BatchNormalization())
        modelA.add(layers.Activation("relu"))
        modelA.add(layers.MaxPooling2D((2, 2)))
        modelA.add(layers.Conv2D(64, (3, 3)))
        modelA.add(layers.BatchNormalization())
        modelA.add(layers.Activation("relu"))
        modelA.add(layers.MaxPooling2D((2, 2)))
        modelA.add(layers.Conv2D(128, (3, 3)))
        modelA.add(layers.BatchNormalization())
        modelA.add(layers.Activation("relu"))
        modelA.add(layers.MaxPooling2D((2, 2)))
        modelA.add(layers.Conv2D(256, (3, 3)))
        modelA.add(layers.BatchNormalization())
        modelA.add(layers.Activation("relu"))
        modelA.add(layers.MaxPooling2D((2, 2)))
        modelA.add(layers.Flatten())
        modelA.add(layers.Dense(self.batch))
        modelA.add(layers.Reshape((1, self.batch)))
        return modelA
    

    def load_model(self):
        K.clear_session()
        loaded_model = self.modelCreator()
        loaded_model.load_weights("static/model_weights.h5")
        global graph
        graph = tf.get_default_graph()
        return loaded_model

    def create_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (300, 150))
        retval, img = cv2.threshold(img, 0, 255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        img = img / 255.
        img = img.reshape(1, 300, 150, 1)
        return img

    def predict(self, path):
        model = self.load_model()
        image = self.create_img(path)
        with graph.as_default():
            ans = model.predict(image)
        return ans

    def get(self, request):
        form = CustomerForm()
        form2 = VerificationForm()
        form3 = NumberOfForms()
        return render(request, self.template_name, {'form': form, 'ver_code': False, 'form3': form3})

    def post(self, request):
        answer = -1
        reg_code = True
        ver_code = False
        global x
        global nof
        if request.method == 'POST':
            form = CustomerForm(request.POST, request.FILES)
            form2 = VerificationForm(request.POST, request.FILES)
            form3 = NumberOfForms(request.POST)
            if 'register' in request.POST:
                print("In registration")
                if form.is_valid():
                    form.save()
                    form = CustomerForm()
                    return render(request, self.template_name, {
                        'reg_code': reg_code,
                        'form': form,
                        'form3': form3,
                        'form2': form2
                    })
            elif 'nof' in request.POST:
                print("Number of Forms")
                verforms = []
                if form3.is_valid():
                    nof = form3.cleaned_data['number']
                for i in range(0, nof):
                    new_form = VerificationForm(prefix=str(i))
                    verforms.append(new_form)
                print(verforms)
                x = verforms
                return render(request, "Main/verification.html", {
                        'reg_code': reg_code,
                        'ver_code': True,
                        'verforms': verforms,
                    })
            elif 'verify' in request.POST:
                print("Verify")
                answers = []
                for i in range(0, nof):
                    j = x[i]
                    j = VerificationForm(request.POST, request.FILES, prefix=str(i))
                    if j.is_valid():
                        VerificationDetails.objects.all().delete()
                        j.save()
                        print(j.cleaned_data['image'])
                        l = str(i) + "-"
                        #----Changes Start Here----
                        sum = 0
                        avg = 0
                        total = len(Attachment.objects.filter(customerdetails__c_id=j.cleaned_data['c_id']))
                        for image in Attachment.objects.filter(customerdetails__c_id=j.cleaned_data['c_id']):
                            print('Registration :'+image.file.path)
                            print('Verification :'+VerificationDetails.objects.get(c_id=j.cleaned_data['c_id']).image.path)
                            vector_database = self.predict(image.file.path)
                            vector_image = self.predict(VerificationDetails.objects.get(c_id=j.cleaned_data['c_id']).image.path)
                            answer = np.sum(np.square(vector_image - vector_database))
                            sum += answer

                        #vector_image = self.predict(VerificationDetails.objects.get(c_id=j.cleaned_data['c_id']).image.path)
                        #vector_database = self.predict(CustomerDetails.objects.get(c_id=j.cleaned_data['c_id']).image.path)
                        #answer = np.sum(np.square(vector_image - vector_database))

                        avg = sum / total
                        print('Total Sum :' + str(sum))
                        print('Average :' + str(avg))

                        if avg < 1015:
                            answer = "The signature is real."
                        else:
                            answer = "The signature is forged."
                        K.clear_session()
                        answers.append(answer)
                return render(request, "Main/prediction.html", {
                    'answers': answers
                })
        else:
            print("ELSE")
            form = CustomerForm()
            form2 = VerificationForm()
            form3 = NumberOfForms()

        return render(request, self.template_name, {
            'form': form,
            'form2': form2,
            'form3': form3,
            'answer': answer,
            'reg_code': reg_code
        })
