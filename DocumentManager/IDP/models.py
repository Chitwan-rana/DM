# Create your models here.
from django.db import models

class PDFOperationTracker(models.Model):
    OPERATION_CHOICES = [
        ('merge', 'Merge'),
        ('compress', 'Compress'),
        ('split', 'Split'),
        ('searchable', 'Convert to Searchable'),
        ('redact', 'Redact'),
        ('pdf_to_docx', 'PDF to DOCX'),
    ]

    operation_type = models.CharField(max_length=50, choices=OPERATION_CHOICES)
    timestamp = models.DateTimeField(auto_now_add=True)
    filename = models.CharField(max_length=255)
    user_ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=512, null=True, blank=True)
    user = models.ForeignKey('auth.User', on_delete=models.SET_NULL, null=True, blank=True)

