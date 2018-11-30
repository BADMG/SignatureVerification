from django.urls import path
from Main.views import MainView, CustomerFormView, VerifyFormView
from . import views

urlpatterns = [
    path('', MainView.as_view(), name='index'),
    path(r'^customer/submit', CustomerFormView.as_view(), name='question'),
    path(r'^verify/submit', VerifyFormView.as_view(), name='answer'),
]