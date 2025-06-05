"""
URL configuration for DocumentManager project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_ap.viewsp import  2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf import settings
from django.urls import path,include
from DocumentManager.views.index import *
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', Home,name='Home'),
    path('register/', register, name='register'),
    path('login/', user_login, name='login'),
    path('logout/', user_logout, name='logout'),
    path('pipeline/',pipeline,name='pipeline'),
    path('page_not_found/', page_not_found, name='page_not_found'),
    path('', include('DocClassifier.urls')), 
    path('', include('DocDeployment.urls')),
    path('', include('DocExtraction.urls')),
    path('', include('DocPostprocess.urls')),
    path('', include('DocPreprocess.urls')), 
    path('', include('DocValidation.urls')),
    path('', include('DocChat.urls')),
]
    
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

