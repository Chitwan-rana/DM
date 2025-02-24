from django import forms
from .models import Label, Image

class LabelForm(forms.ModelForm):
    class Meta:
        model = Label
        fields = ['name']

class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['label', 'image']