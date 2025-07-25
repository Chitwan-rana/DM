from IDP.models import PDFOperationTracker

def track_pdf_operation(operation_type, filename, request=None):
    tracker = PDFOperationTracker(
        operation_type=operation_type,
        filename=filename
    )
    if request:
        tracker.user_ip = request.META.get('REMOTE_ADDR')
        tracker.user_agent = request.META.get('HTTP_USER_AGENT', '')
        if request.user.is_authenticated:
            tracker.user = request.user
    tracker.save()

