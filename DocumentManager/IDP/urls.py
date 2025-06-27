from django.urls import path
from .views import summarize_pdf,merge_pdfs_view,compress_pdf_view,download_compressed_pdf

urlpatterns = [
     path('summarize-pdf/', summarize_pdf, name='summarize_pdf'),
     path('merge-pdfs/', merge_pdfs_view, name='merge_pdfs_views'),
     path('compress-pdf/', compress_pdf_view, name='compress_pdf'),
     path('download-compressed-pdf/', download_compressed_pdf, name='download_compressed_pdf'),
     # path('compress/', views.compress_file, name='compress_file'),
     # path('generatesearchable/', views.generate_searchable_view, name='generate_searchable_view'),
     # path('merge/', views.merge_files_view, name='merge_files_view'),
     # path('split/', views.split_file, name='split_file'),
     # path('summarization/', views.summarization, name='summarization'),

]