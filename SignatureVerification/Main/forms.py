from django import forms
from multiupload.fields import MultiFileField
from Main.models import CustomerDetails, VerificationDetails, Attachment


class CustomerForm(forms.ModelForm):
    c_id = forms.CharField(label="", widget=forms.TextInput(attrs={'placeholder': 'Customer ID', 'class':'form-control mb-4','title':'Enter Only Numeric values','pattern':'[0-9]+'}))
    c_name = forms.CharField(label="", widget=forms.TextInput(attrs={'placeholder': 'Customer Name', 'class':'form-control mb-4','pattern':'[A-Za-z ]+','title':'Enter Characters Only'}))
    #image = forms.FileField(label="")

    class Meta:
        model = CustomerDetails
        fields = ('c_name', 'c_id') #'image', )

    files = MultiFileField(min_num=1, max_num=5, max_file_size=1024 * 1024 * 5)

    def save(self, commit=True):
        instance = super(CustomerForm, self).save(commit)
        for each in self.cleaned_data['files']:
            Attachment.objects.create(file=each, customerdetails=instance)
        return instance

    def clean(self):
        cleaned_data = self.cleaned_data
        uid = cleaned_data.get('c_id')
        matching_id = CustomerDetails.objects.filter(c_id = uid)
        if self.instance:
            matching_id = matching_id.exclude(pk=self.instance.pk)
        if matching_id.exists():
            msg = u"User ID: %s already exist." % uid
            raise forms.ValidationError(msg)
        else:
            return self.cleaned_data

class NumberOfForms(forms.Form):
    number = forms.IntegerField(label="", widget=forms.TextInput(attrs={'placeholder': 'Number of Forms', 'class':'form-control mb-4','title':'Enter Only Numeric values','pattern':'[0-9]+'}))


class VerificationForm(forms.ModelForm):
    c_id = forms.CharField(label="", widget=forms.TextInput(attrs={'placeholder': 'Customer ID', 'class':'form-control mb-4','title':'Enter Only Numeric values','pattern':'[0-9]+'}))
    image = forms.FileField(label="")

    class Meta:
        model = VerificationDetails
        fields = ('c_id', 'image')



            


