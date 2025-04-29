
#views.py
import logging
import traceback
from django.utils.timezone import now
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, update_session_auth_hash
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.db import connection
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import threading
import os
import json
from django.utils.timezone import now  # helpful for timestamps if needed
from .models import ExamUpload, GradingSettings, GradedAnswer, DrivePDF, ChunkedText
from .drive_utils import download_drive_file, extract_pdf_text, chunk_text
from .gdrive_utils import fetch_drive_pdfs_and_chunk
from django.shortcuts import get_object_or_404
# from .drive_processor import process_drive_pdf as process_drive_pdf_logic

from django.http import JsonResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
import json
import os
import logging, sys, pathlib

# Keep existing pipeline imports
from .drive_pipeline import (
    load_and_partition_drive_documents,
    restructure_all_elements_flat,
    generate_captions_from_memory,
    convert_elements_to_langchain_docs,
    dynamic_chunk_documents,
    ingest_chroma,
    embedding_model,
    handle_fallback_file,  # Make sure this is included
)
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .pdf_extractor import PDFExtractor
import sys, os

from .utils.evaluator import evaluate_exam
# from .utils.evaluator import evaluate_exam   # 1️⃣ add import

LOG_FILE = pathlib.Path(__file__).with_name("grademate.log")

logging.basicConfig(
    level=logging.INFO,                                 # DEBUG if you really need it
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),              # uses the patched UTF-8 stdout
    ],
    force=True,                                         # overwrite previous root config
)

logger = logging.getLogger(__name__)

# Import your models
from .models import (
    ExamUpload, GradingSettings, GradedAnswer, 
    DrivePDF, ChunkedText, ExtractedAnswer
)

# Import your pipeline utilities - make sure these paths match your project structure
from .drive_utils import download_drive_file, extract_pdf_text, chunk_text
from .gdrive_utils import fetch_drive_pdfs_and_chunk

# Import your processing modules
from .drive_processor import process_drive_pdf, process_all_drive_pdfs
from .file_utils import handle_unsupported_file, get_supported_extension



# Global variable to store pipeline progress
pipeline_progress = []

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
    """Handle file uploads for grading."""
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        
        if files:
            for file in files:
                # Save the uploaded file
                exam_upload = ExamUpload.objects.create(
                    user=request.user,
                    file=file
                )
            
            # Clear any previous extraction data
            if 'extracted_data' in request.session:
                del request.session['extracted_data']
            
            # Redirect to the OCR extraction page
            return redirect('ocr_extracted')
    
    return render(request, 'grading.html')



@login_required
def direct_pdf_processor(request):
    """Direct PDF processing page for testing."""
    error_message = None
    success_message = None
    extracted_data = None
    
    if request.method == 'POST' and 'pdf_file' in request.FILES:
        pdf_file = request.FILES['pdf_file']
        
        if not pdf_file.name.lower().endswith('.pdf'):
            error_message = "Please upload a PDF file."
        else:
            try:
                # Initialize the PDF extractor
                from .pdf_extractor import PDFExtractor
                extractor = PDFExtractor()
                
                # Process the PDF
                extracted_data = extractor.extract_questions_and_answers(pdf_file)
                success_message = "PDF processed successfully!"
                
                # Store in session
                request.session['extracted_data'] = extracted_data
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_message = f"Error processing PDF: {str(e)}"
    
    context = {
        'error_message': error_message,
        'success_message': success_message,
        'extracted_data': extracted_data
    }
    
    return render(request, 'direct_pdf_processor.html', context)



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



# Add this to your views.py file or update the existing ocr_extracted function



@login_required
def ocr_extracted(request):
    """
    Simplified OCR extraction view with improved error handling
    """
    logger = logging.getLogger(__name__)
    logger.info("ocr_extracted view called")
    
    try:
        # Check if we already have extracted data in session
        extracted_data = request.session.get("extracted_data")
        
        if not extracted_data:
            # Process the latest uploaded PDF
            latest_upload = ExamUpload.objects.filter(user=request.user).order_by("-uploaded_at").first()
            
            if latest_upload:
                logger.info(f"Processing latest upload: {latest_upload.file.path}")
                
                with open(latest_upload.file.path, "rb") as f:
                    from django.conf import settings
                    extractor = PDFExtractor(api_key=settings.GROQ_API_KEY)
                    extracted_data = extractor.extract_questions_and_answers(f)
                    
                    # Store in session
                    request.session["extracted_data"] = extracted_data
                    logger.info(f"Extracted {len(extracted_data)} questions")
            else:
                logger.warning("No uploaded PDF found")
                extracted_data = []
        
        # Prepare context for template
        context = {
            "extracted_data": extracted_data,
            "questions_count": len(extracted_data) if extracted_data else 0
        }
        
        logger.info(f"Rendering OCR extracted view with {len(extracted_data) if extracted_data else 0} questions")
        return render(request, "ocr_extracted.html", context)
        
    except Exception as e:
        logger.error(f"Error in OCR extraction: {e}")
        logger.error(traceback.format_exc())
        
        # Provide empty data in case of error
        context = {
            "extracted_data": [],
            "error_message": f"Error: {str(e)}"
        }
        
        return render(request, "ocr_extracted.html", context)



# Make a Test Endpoint for Debugging

@login_required
def test_pdf_extraction(request):
    """Test endpoint for PDF extraction debugging."""
    if request.method == 'POST' and 'pdf_file' in request.FILES:
        try:
            pdf_file = request.FILES['pdf_file']
            extractor = PDFExtractor()
            
            # First extract text
            text = extractor.extract_text_from_pdf(pdf_file)
            
            # Reset file pointer
            pdf_file.seek(0)
            
            # Then extract Q&A
            questions = extractor.extract_questions_and_answers(pdf_file)
            
            return JsonResponse({
                'status': 'success',
                'text_length': len(text),
                'text_sample': text[:500] + '...',
                'questions_count': len(questions),
                'questions': questions
            })
        except Exception as e:
            import traceback
            return JsonResponse({
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    return render(request, 'test_extraction.html')



def generate_sample_data():
    """Generate comprehensive sample data for the OCR extracted view."""
    sample_data = []
    for i in range(1, 6):  # Generate 5 sample questions
        sample_data.append({
            "question_no": str(i),
            "question_statement": f"Sample Question {i}: Describe a key concept in detail.",
            "complete_answer": f"This is a sample answer for Question {i}, demonstrating a comprehensive response.",
            "debug_info": "Sample data generated due to extraction failure"
        })
    return sample_data




@login_required
def output(request):
    """
    Display the final grading output with detailed logging and robust error handling
    """
    import json
    import traceback
    
    logger = logging.getLogger(__name__)
    logger.info("[DEBUG] output view")
    
    # Get evaluation data from session with comprehensive handling
    evaluation_json = request.session.get('evaluation_json')
    logger.info(f"Output view - evaluation_json from session: {evaluation_json}")
    
    # Parse and validate the JSON
    if evaluation_json:
        try:
            # Handle different data types
            if isinstance(evaluation_json, str):
                # Parse the JSON string
                evaluation_data = json.loads(evaluation_json)
                logger.info(f"[DEBUG] Successfully parsed JSON string, data type: {type(evaluation_data)}")
            else:
                # Already a Python object, just use it directly
                evaluation_data = evaluation_json
                # Serialize it for the template
                evaluation_json = json.dumps(evaluation_data)
                logger.info(f"[DEBUG] Using pre-parsed data, serialized for template")
            
            # Validate the structure to ensure it has required fields
            if 'evaluations' in evaluation_data:
                logger.info(f"[DEBUG] Found {len(evaluation_data['evaluations'])} evaluations")
                
                # Check for score/similarity_percentage field
                for i, eval_item in enumerate(evaluation_data['evaluations']):
                    # Ensure the correct score field exists
                    if 'score' in eval_item and 'similarity_percentage' not in eval_item:
                        # Keep the existing score field
                        logger.info(f"[DEBUG] Question {i+1} has score field: {eval_item['score']}")
                    elif 'similarity_percentage' in eval_item and 'score' not in eval_item:
                        # Create score field for similarity_percentage
                        eval_item['score'] = eval_item['similarity_percentage']
                        logger.info(f"[DEBUG] Question {i+1} copying similarity_percentage to score: {eval_item['score']}")
                    elif 'score' not in eval_item and 'similarity_percentage' not in eval_item:
                        # Neither field exists, add default
                        eval_item['score'] = 0
                        logger.info(f"[DEBUG] Question {i+1} missing score fields, adding default")
                
                # Update the JSON string with possibly modified data
                evaluation_json = json.dumps(evaluation_data)
            elif 'total_score' in evaluation_data:
                logger.info(f"[DEBUG] Found total_score: {evaluation_data['total_score']}")
            else:
                logger.warning("[DEBUG] Missing expected structure in evaluation data")
                
        except json.JSONDecodeError as e:
            logger.error(f"[DEBUG] JSON parsing error: {e}")
            logger.error(traceback.format_exc())
            evaluation_data = None
            evaluation_json = json.dumps({
                "error": f"Invalid JSON data: {str(e)}",
                "data_sample": str(evaluation_json)[:100] if evaluation_json else "None"
            })
        except Exception as e:
            logger.error(f"[DEBUG] Unexpected error processing evaluation data: {e}")
            logger.error(traceback.format_exc())
            evaluation_data = None
            evaluation_json = json.dumps({
                "error": f"Error processing evaluation data: {str(e)}"
            })
    else:
        logger.warning("[DEBUG] No evaluation_json found in session")
        evaluation_data = None
        evaluation_json = json.dumps({
            "error": "No evaluation data available",
            "message": "Please grade an exam first."
        })
    
    # Log final data being sent to template
    logger.info(f"[DEBUG] Final evaluation_json for template (first 200 chars): {evaluation_json[:200]}...")
    
    # Prepare context for template
    context = {
        'evaluation_json': evaluation_json,
        'debug_info': {
            'has_evaluation_data': evaluation_data is not None,
            'data_type': type(evaluation_data).__name__ if evaluation_data else 'None',
            'session_keys': list(request.session.keys()),
            'evaluation_summary': f"{len(evaluation_data.get('evaluations', []))} questions" if evaluation_data and 'evaluations' in evaluation_data else 'No evaluation data'
        }
    }
    
    logger.info(f"[DEBUG] Rendering output template with context: {context}")
    
    return render(request, 'output.html', context)




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
def fetch_drive_pdfs_view(request):
    if request.method == 'POST':
        logs = []
        try:
            print("[DEBUG] fetch_drive_pdfs_view triggered by:", request.user)
            logs.append("🚀 Starting fetch process on backend...")

            fetched = fetch_drive_pdfs_and_chunk(request.user, logs)

            print("[DEBUG] Number of PDFChunk objects returned:", len(fetched))
            files = [{"id": obj.id, "title": obj.title} for obj in fetched]

            print("[DEBUG] JSON files to return:", files)
            return JsonResponse({
                "status": "ok",
                "logs": logs,
                "files": files
            })

        except Exception as e:
            logs.append(f"❌ Unexpected error: {str(e)}")
            print("[ERROR] Exception in fetch_drive_pdfs_view:", e)
            return JsonResponse({
                "status": "error",
                "logs": logs,
                "error": str(e)
            })


@login_required
def process_drive_pipeline(request):
    """
    Process all downloaded Drive files for the current user
    """
    from .drive_processor import process_all_drive_pdfs
    
    logs = []
    
    def progress_callback(message):
        logs.append(message)
    
    try:
        # Call the simplified processor that doesn't depend on BLIP
        results = process_all_drive_pdfs(
            user=request.user,
            progress_callback=progress_callback
        )
        
        successful_files = results.get('successful_files', 0)
        failed_files = results.get('failed_files', [])
        
        status = "ok" if successful_files > 0 or results.get('status') == "no_files" else "error"
        message = f"✅ Completed Drive Processing: {successful_files} files successful, {len(failed_files)} files failed."
        
        if failed_files:
            failed_titles = [f.get('title', 'Unknown') for f in failed_files]
        else:
            failed_titles = []
        
        return JsonResponse({
            "status": status,
            "timestamp": now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": message,
            "logs": logs,
            "success_count": successful_files,
            "failed_count": len(failed_files),
            "failed_files": failed_titles
        })
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logs.append(f"❌ Processing pipeline failed with error: {str(e)}")
        logs.append(error_traceback)
        
        return JsonResponse({
            "status": "error",
            "timestamp": now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": f"Pipeline failed: {str(e)}",
            "logs": logs,
            "success_count": 0,
            "failed_count": 0,
            "failed_files": []
        })
    

@login_required
def drive_pipeline_page(request):
    return render(request, 'drive_pipeline.html')

@login_required
@csrf_exempt
def start_drive_pipeline(request):
    global pipeline_progress
    pipeline_progress = []

    def pipeline_runner(file_ids):
        from app.drive_pipeline import main as drive_main  # Import here to avoid circular imports
        try:
            pipeline_progress.append("🚀 Starting Drive pipeline...")

            drive_main(file_ids=file_ids, progress_callback=lambda msg: pipeline_progress.append(msg))

            pipeline_progress.append("✅ Drive pipeline completed successfully!")
        except Exception as e:
            pipeline_progress.append(f"❌ Error: {str(e)}")

    try:
        # Get the latest DrivePDFs for the user
        drive_pdfs = DrivePDF.objects.filter(user=request.user)
        if not drive_pdfs.exists():
            return JsonResponse({'status': 'error', 'message': 'No Drive PDFs found.'})

        file_ids = [pdf.drive_file_id for pdf in drive_pdfs]

        # Run the pipeline in a new thread
        threading.Thread(target=pipeline_runner, args=(file_ids,)).start()

        return JsonResponse({'status': 'started'})

    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def get_drive_pipeline_progress(request):
    global pipeline_progress
    return JsonResponse({'progress': pipeline_progress})



@login_required
def get_processed_files(request):
    """API endpoint to get all processed files"""
    try:
        drive_pdfs = DrivePDF.objects.filter(user=request.user)
        files_data = []
        
        for pdf in drive_pdfs:
            chunk_count = ChunkedText.objects.filter(pdf=pdf).count()
            files_data.append({
                'id': pdf.id,
                'title': pdf.title,
                'drive_file_id': pdf.drive_file_id,
                'chunk_count': chunk_count,
                'uploaded_at': pdf.uploaded_at.strftime('%Y-%m-%d %H:%M')
            })
        
        return JsonResponse({
            'status': 'success',
            'files': files_data
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)



@login_required
def get_file_chunks(request):
    """API endpoint to get chunks for a specific file"""
    try:
        file_id = request.GET.get('file_id')
        if not file_id:
            return JsonResponse({
                'status': 'error',
                'error': 'No file ID provided'
            }, status=400)
        
        # Fetch the DrivePDF object, ensuring it belongs to the current user
        pdf = get_object_or_404(DrivePDF, id=file_id, user=request.user)

        # Fetch the associated chunks
        chunks = ChunkedText.objects.filter(pdf=pdf).order_by('order')
        
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                'id': chunk.id,
                'order': chunk.order,
                'content': chunk.content
            })
        
        return JsonResponse({
            'status': 'success',
            'file_id': pdf.id,
            'file_title': pdf.title,
            'chunks': chunks_data
        })
    except Exception as e:
        logger.error(f"Error fetching chunks for file ID {file_id}: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'error': 'Internal Server Error',
            'message': str(e)  # Log the error message in the response for debugging
        }, status=500)






@login_required
@csrf_exempt
def upload_exam(request):
    """Handle exam paper upload, extract questions and answers."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method allowed'}, status=405)
    
    if 'file' not in request.FILES:
        return JsonResponse({'status': 'error', 'message': 'No file uploaded'}, status=400)
    
    file = request.FILES['file']
    
    # Check if file is a PDF
    if not file.name.lower().endswith('.pdf'):
        return JsonResponse({'status': 'error', 'message': 'Only PDF files are supported'}, status=400)
    
    try:
        # Create a temporary path to store the file
        temp_path = f"temp_exams/{request.user.id}_{file.name}"
        path = default_storage.save(temp_path, ContentFile(file.read()))
        
        # Reset file pointer to start
        file.seek(0)


        # Initialize the extractor
        groq_api_key = os.environ.get('GROQ_API_KEY')
        extractor = PDFExtractor(api_key=groq_api_key)
        
        # First extract all text
        text_extraction_result = {
            'status': 'processing',
            'message': 'Extracting text from PDF...',
            'extracted_text': None
        }
        
        # Return initial response for long-running process
        return JsonResponse(text_extraction_result)
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Error processing PDF: {str(e)}'
        }, status=500)




@login_required
def process_exam(request):
    try:
        data = json.loads(request.body)
        file_path = data.get('file_path')
        
        if not file_path:
            return JsonResponse({
                'status': 'error', 
                'message': 'No file path provided'
            }, status=400)
        
        # Check file exists and is a PDF
        if not default_storage.exists(file_path) or not file_path.lower().endswith('.pdf'):
            return JsonResponse({
                'status': 'error', 
                'message': 'Invalid or missing PDF file'
            }, status=404)
        
        with default_storage.open(file_path, 'rb') as f:
            extractor = PDFExtractor()
            questions_and_answers = extractor.extract_questions_and_answers(f)
            
            if not questions_and_answers:
                return JsonResponse({
                    'status': 'error', 
                    'message': 'Could not extract exam data'
                }, status=500)
            
            return JsonResponse({
                'status': 'success',
                'questions_and_answers': questions_and_answers
            })
            
    except Exception as e:
        logger.exception("Exam processing error")
        return JsonResponse({
            'status': 'error', 
            'message': f'Unexpected error: {str(e)}'
        }, status=500)




@login_required
def exam_extraction_status(request, task_id):
    """Check the status of a long-running PDF extraction task."""
    # This would be implemented with a task queue like Celery
    # For now, return a placeholder
    return JsonResponse({
        'status': 'completed',
        'message': 'PDF processing completed',
        'result': None  # Would contain the actual result
    })




@login_required
@csrf_exempt
def grade_exam(request):
    """
    Handle grading of an already extracted exam with robust JSON parsing
    """
    logger = logging.getLogger(__name__)
    logger.info("grade_exam view called")

    # Only allow POST requests
    if request.method != 'POST':
        return JsonResponse({
            'status': 'error', 
            'error': 'Only POST method allowed'
        }, status=405)

    try:
        # Parse request body with robust error handling
        try:
            data = json.loads(request.body)
            extracted_data = data.get('extracted_data', [])
            
            # Validate data is in the right format
            if not isinstance(extracted_data, list):
                logger.error(f"Invalid data format: {type(extracted_data)}")
                return JsonResponse({
                    'status': 'error',
                    'error': 'Invalid data format'
                }, status=400)
                
            logger.info(f"Received {len(extracted_data)} questions for grading")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return JsonResponse({
                'status': 'error',
                'error': f'Invalid JSON: {str(e)}'
            }, status=400)

        # Process grading
        try:
            # Import the evaluation function
            from .utils.evaluator import evaluate_exam
            
            # Evaluate the exam
            evaluation_result = evaluate_exam(extracted_data)
            
            # Convert the result to a dictionary
            if hasattr(evaluation_result, 'model_dump'):
                result_dict = evaluation_result.model_dump()
            elif hasattr(evaluation_result, 'dict'):
                result_dict = evaluation_result.dict()
            else:
                result_dict = evaluation_result
                
            # Convert to JSON - use Python's json module
            evaluation_json = json.dumps(result_dict)
            
            # Store in session
            request.session['evaluation_json'] = evaluation_json
            
            # Return the result
            return JsonResponse({
                'status': 'success',
                'evaluation_json': evaluation_json,
                'total_questions': len(extracted_data)
            })
            
        except Exception as e:
            logger.error(f"Grading error: {e}")
            logger.error(traceback.format_exc())
            return JsonResponse({
                'status': 'error',
                'error': f'Grading error: {str(e)}'
            }, status=500)
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            'status': 'error',
            'error': f'Server error: {str(e)}'
        }, status=500)