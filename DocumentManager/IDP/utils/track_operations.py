from IDP.models import PDFOperationTracker

def track_pdf_operation(operation_type, filename):
    PDFOperationTracker.objects.create(operation_type=operation_type, filename=filename)
