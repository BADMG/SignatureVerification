from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.generic import TemplateView
from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from Main.forms import CustomerForm, VerificationForm
from Main.models import CustomerDetails, VerificationDetails
import numpy as np
from keras import backend as K
import keras.models as models
import tensorflow as tf
import keras.layers as layers
from PIL import Image


class MainView(TemplateView):
    template_name = 'Main/index.html'
    img_width = 300
    img_height = 150
    batch = 128

    def modelCreator(self):
        modelA = models.Sequential()
        modelA.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(self.img_width, self.img_height, 1)))
        modelA.add(layers.MaxPooling2D((2, 2)))
        modelA.add(layers.Dropout(0.2))
        modelA.add(layers.Conv2D(32, (3, 3), activation='relu'))
        modelA.add(layers.MaxPooling2D((2, 2)))
        modelA.add(layers.Conv2D(64, (3, 3), activation='relu'))
        modelA.add(layers.MaxPooling2D((2, 2)))
        modelA.add(layers.Conv2D(128, (3, 3), activation='relu'))
        modelA.add(layers.MaxPooling2D((2, 2)))
        modelA.add(layers.Flatten())
        modelA.add(layers.Dense(self.batch))
        modelA.add(layers.Reshape((1, self.batch)))
        return modelA

    def load_model(self):
        loaded_model = self.modelCreator()
        loaded_model.load_weights("static/model_weights.h5")
        global graph
        graph = tf.get_default_graph()
        return loaded_model

    def create_img(self, path):
        img = Image.open(path).convert('L')
        img = np.array(img)
        img = img.reshape(1, 300, 150, 1)
        img = img / 255.
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
        return render(request, self.template_name, {'form': form, 'form2': form2})

    def post(self, request):
        answer = -1
        if request.method == 'POST':
            form = CustomerForm(request.POST, request.FILES)
            form2 = VerificationForm(request.POST, request.FILES)
            if 'register' in request.POST:
                if form.is_valid():
                    form.save()
                    return render(request, self.template_name, {
                        'form': form,
                        'form2': form2
                    })
            else:
                if form2.is_valid():
                    VerificationDetails.objects.all().delete()
                    init = form2.save()
                    init.save()
                    vector_image = self.predict(VerificationDetails.objects.get(c_id=form2.cleaned_data['c_id']).image.path)
                    vector_database = self.predict(CustomerDetails.objects.get(c_id=form2.cleaned_data['c_id']).image.path)
                    answer = np.sum(np.square(vector_image - vector_database))
                    K.clear_session()
                    return render(request, "Main/prediction.html", {
                        'answer': answer
                    })
        else:
            form = CustomerForm()
            form2 = VerificationForm()
        return render(request, self.template_name, {
            'form': form,
            'form2': form2,
            'answer': answer
        })
