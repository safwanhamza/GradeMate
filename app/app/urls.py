from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from .views import *
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
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),  

    path('dashboard/', dashboard, name='dashboard'),
    path('acc-settings/', acc_settings, name='acc_settings'),
    path('grading/', grading, name='grading'),
    path('grad-settings/', grad_settings, name='grad_settings'),
    path('ocr-extracted/', ocr_extracted, name='ocr_extracted'),
    path('output/', output, name='output'),

    # --- Existing Drive APIs ---
    path('api/upload-drive-pdf/', upload_and_chunk_drive_file, name='upload_drive_pdf'),
    path('fetch-drive-pdfs/', fetch_drive_pdfs_view, name='fetch_drive_pdfs'),  # Fixed duplicate
    path('process-drive-pipeline/', process_drive_pipeline, name='process_drive_pipeline'),

    # --- 🆕 New Drive Pipeline API ---
    path('drive-pipeline/', drive_pipeline_page, name='drive_pipeline_page'),
    path('start-drive-pipeline/', start_drive_pipeline, name='start_drive_pipeline'),
    path('get-drive-pipeline-progress/', get_drive_pipeline_progress, name='get_drive_pipeline_progress'),
    path('get-processed-files/', get_processed_files, name='get_processed_files'),
    path('get-file-chunks/', get_file_chunks, name='get_file_chunks'),
    # Add to urls.py
    path('upload-exam/', upload_exam, name='upload_exam'),
    # Add this to your urlpatterns list in urls.py
    path('grade-exam/', grade_exam, name='grade_exam'),
    
    path('process-exam/', process_exam, name='process_exam'),
    path('extraction-status/<str:task_id>/', exam_extraction_status, name='exam_extraction_status'),
    path('direct-pdf-processor/', direct_pdf_processor, name='direct_pdf_processor'),
    path('admin/', admin.site.urls),
    # New PDF download endpoint
    path('download-report/', download_report_pdf, name='download_report_pdf'),
]


