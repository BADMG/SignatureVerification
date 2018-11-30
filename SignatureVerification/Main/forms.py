from django import forms
from Main.models import CustomerDetails, VerificationDetails


class CustomerForm(forms.ModelForm):
    c_id = forms.CharField(label="", widget=forms.TextInput(attrs={'placeholder': 'Customer ID', 'class':'form-control mb-4'}))
    c_name = forms.CharField(label="", widget=forms.TextInput(attrs={'placeholder': 'Customer Name', 'class':'form-control mb-4'}))
    image = forms.FileField(label="")

    class Meta:
        model = CustomerDetails
        fields = ('c_name', 'c_id', 'image', )


class VerificationForm(forms.ModelForm):
    c_id = forms.CharField(label="", widget=forms.TextInput(attrs={'placeholder': 'Customer ID', 'class':'form-control mb-4'}))
    image = forms.FileField(label="")

    class Meta:
        model = VerificationDetails
        fields = ('c_id', 'image')

