from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from .views import *
"""
URL configuration for app project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""


urlpatterns = [
    path('', home, name='home'),
    path('login-signup/', login_signup, name='login_signup'),
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),  # ✅ Added logout
    path('dashboard/', dashboard, name='dashboard'),
    path('acc-settings/', acc_settings, name='acc_settings'),
    path('grading/', grading, name='grading'),
    path('grad-settings/', grad_settings, name='grad_settings'),
    path('ocr-extracted/', ocr_extracted, name='ocr_extracted'),
    path('output/', output, name='output'),
    path('admin/', admin.site.urls),
    path('api/upload-drive-pdf/', upload_and_chunk_drive_file, name='upload_drive_pdf'),
    path('fetch-drive-pdfs/', fetch_drive_pdfs, name='fetch_drive_pdfs'),

]


# comment here