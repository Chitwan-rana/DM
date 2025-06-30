from django.urls import path
from .views import document_chat_view

urlpatterns = [
    path('Chat/', document_chat_view, name="document_chat"),
]
