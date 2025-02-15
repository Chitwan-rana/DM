from django import forms
from .models import UploadedPDF

class UploadPDFForm(forms.ModelForm):
    class Meta:
        model = UploadedPDF
        fields = ['file']

class QuestionForm(forms.Form):
    question = forms.CharField(label="Enter Your Question", max_length=500)
