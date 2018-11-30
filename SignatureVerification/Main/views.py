from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.generic import TemplateView
from django.shortcuts import render, redirect
from Main.forms import CustomerForm
from Main.models import CustomerDetails

class MainView(TemplateView):
    template_name = 'Main/index.html'

    def get(self, request):
        form = CustomerForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        if request.method == 'POST':
            form = CustomerForm(request.POST, request.FILES)
            if form.is_valid():
                form.save()
                form = CustomerForm()
        else:
            form = CustomerForm()
        return render(request, self.template_name, {
            'form': form
        })



