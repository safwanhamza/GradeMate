from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from .models import ExamUpload, GradingSettings, GradedAnswer
from django.db import connection
from django.contrib.auth import update_session_auth_hash
from django.http import JsonResponse
from .models import DrivePDF, ChunkedText
from .drive_utils import download_drive_file, extract_pdf_text, chunk_text


def home(request):
    print("[DEBUG] home view loaded")
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'index.html')

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

    # Debug raw table check
    with connection.cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("[DEBUG] Tables in database:", tables)

    try:
        papers_graded = GradedAnswer.objects.filter(extracted__exam__user=user).count()
    except Exception as e:
        print("[ERROR] GradedAnswer query failed:", e)
        papers_graded = 0

    try:
        books_inputted = GradingSettings.objects.filter(user=user).count()
    except Exception as e:
        print("[ERROR] GradingSettings query failed:", e)
        books_inputted = 0

    try:
        active_exams = ExamUpload.objects.filter(user=user).count()
    except Exception as e:
        print("[ERROR] ExamUpload query failed:", e)
        active_exams = 0

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
    if request.method == 'POST':
        action = request.POST.get('action')
        value = request.POST.get('value')

        try:
            if action == 'name':
                request.user.first_name = value
                request.user.save()
                return JsonResponse({'status': 'success', 'message': 'Name updated successfully!'})

            elif action == 'email':
                if User.objects.filter(email=value).exclude(pk=request.user.pk).exists():
                    return JsonResponse({'status': 'error', 'message': 'Email already in use!'})
                request.user.email = value
                request.user.username = value  # if username is also email
                request.user.save()
                return JsonResponse({'status': 'success', 'message': 'Email updated successfully!'})

            elif action == 'password':
                request.user.set_password(value)
                request.user.save()
                update_session_auth_hash(request, request.user)  # Keep user logged in
                return JsonResponse({'status': 'success', 'message': 'Password updated successfully!'})

            elif action == 'delete':
                request.user.delete()
                return JsonResponse({'status': 'success', 'message': 'Account deleted successfully!', 'redirect': True})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

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

    # Fetch previously uploaded Drive PDFs for dropdown
    drive_pdfs = DrivePDF.objects.filter(user=request.user)

    if request.method == 'POST':
        print("[DEBUG] POST request in grad_settings")

        # Handle either uploaded textbook or Drive selection
        textbook = request.FILES.get('textbook')
        selected_pdf_id = request.POST.get('selected_pdf')

        if not textbook and selected_pdf_id:
            # If no file uploaded, use existing DrivePDF as context
            selected_pdf = DrivePDF.objects.get(id=selected_pdf_id, user=request.user)
            textbook = selected_pdf.title  # Or store reference

        GradingSettings.objects.create(
            user=request.user,
            textbook=textbook,
            key=request.FILES.get('key'),
            similarity=request.POST.get('similarity', 1.0),
            creativity=request.POST.get('creativity', 0.5),
            manual_text=request.POST.get('manual_text', ''),
            context_choice=request.POST.get('context', 'textbook')
        )

        print("[DEBUG] GradingSettings saved for:", request.user)
        return redirect('ocr_extracted')

    return render(request, 'grad_settings.html', {'drive_pdfs': drive_pdfs})

@login_required
def ocr_extracted(request):
    print("[DEBUG] ocr_extracted view")
    return render(request, 'ocr_extracted.html')

@login_required
def output(request):
    print("[DEBUG] output view")
    return render(request, 'output.html')



@login_required
def upload_and_chunk_drive_file(request):
    if request.method == 'POST':
        file_id = request.POST.get('file_id')
        title = request.POST.get('title', 'Untitled')

        pdf_stream = download_drive_file(file_id)
        full_text = extract_pdf_text(pdf_stream)
        chunks = chunk_text(full_text)

        drive_pdf = DrivePDF.objects.create(
            user=request.user,
            title=title,
            drive_file_id=file_id
        )

        for i, chunk in enumerate(chunks):
            ChunkedText.objects.create(pdf=drive_pdf, content=chunk, order=i)

        return JsonResponse({'message': f'{len(chunks)} chunks created.', 'pdf_id': drive_pdf.id})
    return JsonResponse({'error': 'Invalid method'}, status=400)


@login_required
def fetch_drive_pdfs(request):
    print("[DEBUG] fetch_drive_pdfs view")
    logs = []

    try:
        # Your existing Drive & chunk logic...
        fetched_files = [...]  # List of filenames

        for file in fetched_files:
            logs.append(f"📁 {file} fetched successfully")
            logs.append(f"✅ Chunks created for {file}")

        return JsonResponse({"status": "ok", "logs": logs})
    
    except Exception as e:
        logs.append(str(e))
        return JsonResponse({"status": "error", "error": str(e), "logs": logs})