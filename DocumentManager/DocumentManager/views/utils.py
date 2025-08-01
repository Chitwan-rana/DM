from django.utils import timezone
from django.db.models import Count
from django.db.models.functions import TruncDate, TruncWeek
from IDP.models import PDFOperationTracker
from datetime import timedelta
import json
# Use utils.py to encapsulate the logic for fetching PDF stats into index.html 
def get_pdf_stats(filter_type='all'):
    now = timezone.now()

    # Determine start date based on filter
    if filter_type == 'today':
        start_date = now.replace(hour=0, minute=0, second=0)
    elif filter_type == 'week':
        start_date = now - timedelta(days=7)
    elif filter_type == 'month':
        start_date = now - timedelta(days=30)
    else:
        start_date = None

    # Query the operations
    operations = PDFOperationTracker.objects.filter(timestamp__gte=start_date) if start_date else PDFOperationTracker.objects.all()

    # Total documents
    total_documents = operations.count()

    # Operation type stats
    stats_by_type = (
        operations.values('operation_type')
        .annotate(count=Count('id'))
        .order_by('-count')
    )

    # Convert operation codes to labels
    operation_display = dict(PDFOperationTracker.OPERATION_CHOICES)
    for stat in stats_by_type:
        stat['operation_label'] = operation_display.get(stat['operation_type'], stat['operation_type'])

    # Trend grouping
    if filter_type == 'month':
        grouped = operations.annotate(period=TruncWeek('timestamp'))
    else:
        grouped = operations.annotate(period=TruncDate('timestamp'))

    # Count by period
    date_counts = grouped.values('period').annotate(count=Count('id')).order_by('period')

    trend_labels = [entry['period'].strftime('%Y-%m-%d') for entry in date_counts]
    trend_counts = [entry['count'] for entry in date_counts]

    # Get recent 10 operations
    recent_operations = operations.order_by('-timestamp')[:10]

    # Final context to return
    return {
        'total_documents': total_documents,
        'stats_by_type': stats_by_type,
        'operation_labels': [s['operation_label'] for s in stats_by_type],
        'operation_counts': [s['count'] for s in stats_by_type],
        'trend_labels': json.dumps(trend_labels),
        'trend_counts': json.dumps(trend_counts),
        'recent_operations': recent_operations,
        'filter_type': filter_type,
    }
