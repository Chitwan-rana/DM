from django.db import models

class Label(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name

class Image(models.Model):
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images/')

    def __str__(self):
        return f"{self.label.name} - {self.image.name}"