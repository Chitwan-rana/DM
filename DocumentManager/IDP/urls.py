from django.urls import path
from .views import merge_pdfs_view,compress_pdf_view,download_compressed_pdf,split_pdf_view,download_split_pdf,convert_pdf,download_searchable_pdf,redact_confirm,redact

urlpatterns = [
     path('merge-pdfs/', merge_pdfs_view, name='merge_pdfs_views'),
     path('compress-pdf/', compress_pdf_view, name='compress_pdf'),
     path('download-compressed-pdf/', download_compressed_pdf, name='download_compressed_pdf'),
     path('split-pdf/', split_pdf_view, name='split_pdf'),
     path('download-split-pdf/', download_split_pdf, name='download_split_pdf'),
     path('convert/', convert_pdf, name='convert_pdf'),
     path('download-searchable-pdf/',download_searchable_pdf, name='download_searchable_pdf'),
     path('redact/', redact, name='redact'),
     path('redact-confirm/', redact_confirm, name='redact_confirm'),


]