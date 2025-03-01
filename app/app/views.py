from django.shortcuts import render

def home(request):
    return render(request, 'index.html')

def login_signup(request):
    return render(request, 'login_signup.html')

def dashboard(request):
    return render(request, 'dashboard.html')

def acc_settings(request):
    return render(request, 'acc_settings.html')

def grading(request):
    return render(request, 'grading.html')

def grad_settings(request):
    return render(request, 'grad_settings.html')

def ocr_extracted(request):
    return render(request, 'ocr_extracted.html')

def output(request):
    return render(request, 'output.html')