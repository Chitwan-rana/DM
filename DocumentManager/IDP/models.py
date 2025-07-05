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
