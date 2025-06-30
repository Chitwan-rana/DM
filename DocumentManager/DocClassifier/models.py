# models.py
from django.db import models

class Label(models.Model):
    name = models.CharField(max_length=100)

class Image(models.Model):
    image = models.ImageField(upload_to='images/')  # Ensure this is correctly defined
    label = models.ForeignKey(Label, on_delete=models.CASCADE)