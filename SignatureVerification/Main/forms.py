from django import forms
from multiupload.fields import MultiFileField
from Main.models import CustomerDetails, VerificationDetails, Attachment


class CustomerForm(forms.ModelForm):
    c_id = forms.CharField(label="", widget=forms.TextInput(attrs={'placeholder': 'Customer ID', 'class':'form-control mb-4'}))
    c_name = forms.CharField(label="", widget=forms.TextInput(attrs={'placeholder': 'Customer Name', 'class':'form-control mb-4'}))

    class Meta:
        model = CustomerDetails
        fields = ('c_name', 'c_id')
    
    files = MultiFileField(min_num=1, max_num=5, max_file_size=1024*1024*5)
    
    def save(self, commit=True):
        instance = super(CustomerForm, self).save(commit)
        for each in self.cleaned_data['files']:
            Attachment.objects.create(file=each, customerdetails=instance)
        return instance


class VerificationForm(forms.ModelForm):
    c_id = forms.CharField(label="", widget=forms.TextInput(attrs={'placeholder': 'Customer ID', 'class':'form-control mb-4'}))
    image = forms.FileField(label="")

    class Meta:
        model = VerificationDetails
        fields = ('c_id', 'image')

