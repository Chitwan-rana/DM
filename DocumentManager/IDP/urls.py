from django.urls import path
from .views import merge_pdfs_view,compress_pdf_view,download_compressed_pdf,split_pdf_view,download_split_pdf,convert_pdf,download_searchable_pdf,redact_confirm,redact,upload_pdf,convert_pdf_to_docx,convert_pdf_view,download_split_pdf_view

urlpatterns = [
     # PDF MERGE
     path('merge-pdfs/', merge_pdfs_view, name='merge_pdfs_views'),
     # PDF COMPRESSION
     path('compress-pdf/', compress_pdf_view, name='compress_pdf'),
     path('download-compressed-pdf/', download_compressed_pdf, name='download_compressed_pdf'),
     # PDF SPLIT
     path('split-pdf/', split_pdf_view, name='split_pdf'),
     path('download-split-pdf/', download_split_pdf, name='download_split_pdf'),
     path('split/result/', download_split_pdf_view, name='download_split_pdf_page'),
     # PDF SEARCHABLE
     path('convert/', convert_pdf, name='convert_pdf'),
     path('download-searchable-pdf/',download_searchable_pdf, name='download_searchable_pdf'),
     # REDACTION
     path('redact/', redact, name='redact'),
     path('redact-confirm/', redact_confirm, name='redact_confirm'),
     # PDF TO DOCX
     path('convert-pdf-view/', convert_pdf_view, name='convert_pdf_view'),
     path('upload/', upload_pdf, name='upload_pdf'),
     path('pdf_to_docx/', convert_pdf_to_docx, name='convert_pdf_to_docx'),


]