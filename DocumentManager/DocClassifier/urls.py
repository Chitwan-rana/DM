from django.urls import path
from . import views
from .views import reset_data

urlpatterns = [
    path('classify/', views.classify, name='classify'),
    path('train/', views.train_model, name='train_model'),
    path('predict/', views.predict_label, name='predict_label'),
    path('reset/', reset_data, name='reset_data'),
]