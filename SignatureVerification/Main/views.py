from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.generic import TemplateView
from django.shortcuts import render, redirect
from Main.forms import CustomerForm, VerificationForm
from Main.models import CustomerDetails
from PIL import Image
import numpy as np
from keras.models import model_from_json


class MainView(TemplateView):
    template_name = 'Main/index.html'

    def load_model(self):
        json_file = open('static/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("static/model_file.h5")
        return loaded_model

    def create_img(self, path):
        im = Image.open(path).convert('RGB')
        im = im.thumbnail((300, 150), Image.ANTIALIAS)
        im = np.array(im)
        im = im / 255
        # Any other preprocessing
        return im

    def predict(self, path):
        model = load_model()
        image = create_img(path)
        ans = model.predict(image)
        return ans

    def get(self, request):
        form = CustomerForm()
        form2 = VerificationForm()
        return render(request, self.template_name, {'form': form, 'form2': form2})

    def post(self, request):
        answer = 0
        if request.method == 'POST':
            form = CustomerForm(request.POST, request.FILES)
            form2 = VerificationForm(request.POST, request.FILES)
            if form.is_valid() or form2.is_valid():
                form.save()
                form = CustomerForm()
                form2.save()
                form2 = VerificationForm()
        else:
            form = CustomerForm()
            form2 = VerificationForm()
        return render(request, self.template_name, {
            'form': form,
            'form2': form2,
            'answer': answer
        })

