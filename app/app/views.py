from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from .models import ExamUpload, GradingSettings, GradedAnswer


@login_required
def home(request):
    print("[DEBUG] home view loaded")
    return render(request, 'index.html')

@login_required
def login_signup(request):
    print("[DEBUG] login_signup view triggered")
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        print("[DEBUG] POST data received:", email)

        if 'signup' in request.POST:
            if User.objects.filter(username=email).exists():
                print("[DEBUG] Email already exists:", email)
                return render(request, 'login_signup.html', {'error': 'Email already exists'})
            user = User.objects.create_user(username=email, email=email, password=password)
            login(request, user)
            print("[DEBUG] New user signed up and logged in:", email)
            return redirect('dashboard')
        else:
            user = authenticate(request, username=email, password=password)
            if user:
                login(request, user)
                print("[DEBUG] User logged in:", email)
                return redirect('dashboard')
            else:
                print("[DEBUG] Invalid login credentials")
                return render(request, 'login_signup.html', {'error': 'Invalid credentials'})
    return render(request, 'login_signup.html')

@login_required
def dashboard(request):
    user = request.user
    print("[DEBUG] dashboard view for user:", user)

    papers_graded = GradedAnswer.objects.filter(extracted__exam__user=user).count()
    books_inputted = GradingSettings.objects.filter(user=user).count()
    active_exams = ExamUpload.objects.filter(user=user).count()

    print("[DEBUG] papers_graded:", papers_graded)
    print("[DEBUG] books_inputted:", books_inputted)
    print("[DEBUG] active_exams:", active_exams)

    context = {
        'papers_graded': papers_graded,
        'books_inputted': books_inputted,
        'active_exams': active_exams,
    }
    return render(request, 'dashboard.html', context)

@login_required
def acc_settings(request):
    print("[DEBUG] acc_settings view")
    return render(request, 'acc_settings.html')

@login_required
def grading(request):
    print("[DEBUG] grading view triggered")
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        print("[DEBUG] Files received:", files)
        if files:
            for file in files:
                print(f"[DEBUG] Saving file: {file.name}")
                ExamUpload.objects.create(user=request.user, file=file)
            return redirect('ocr_extracted')
        else:
            print("[DEBUG] No files uploaded")
    return render(request, 'grading.html')

@login_required
def grad_settings(request):
    print("[DEBUG] grad_settings view")
    if request.method == 'POST':
        print("[DEBUG] POST request in grad_settings")
        GradingSettings.objects.create(
            user=request.user,
            textbook=request.FILES['textbook'],
            key=request.FILES['key'],
            similarity=request.POST['similarity'],
            creativity=request.POST['creativity'],
            manual_text=request.POST.get('manual_text', ''),
            context_choice=request.POST['context']
        )
        print("[DEBUG] GradingSettings saved for:", request.user)
        return redirect('ocr_extracted')
    return render(request, 'grad_settings.html')

@login_required
def ocr_extracted(request):
    print("[DEBUG] ocr_extracted view")
    return render(request, 'ocr_extracted.html')

@login_required
def output(request):
    print("[DEBUG] output view")
    return render(request, 'output.html')
