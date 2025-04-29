import io
from typing import List, Dict, Any, Optional
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
import json
import os
import re

def clean_html(text):
    """Remove HTML tags from text to avoid reportlab errors."""
    if not text:
        return ""
    return re.sub(r'<[^>]+>', '', text)

def convert_evaluation_to_dict(evaluation):
    """
    Convert an ExamEvaluation object to a dictionary, handling both Pydantic objects and dictionaries.
    """
    if hasattr(evaluation, 'model_dump'):
        # Pydantic v2
        return evaluation.model_dump()
    elif hasattr(evaluation, 'dict'):
        # Pydantic v1
        return evaluation.dict()
    elif isinstance(evaluation, dict):
        # Already a dict
        return evaluation
    else:
        # Handle other types appropriately
        return {"error": "Unknown evaluation type", "evaluations": []}

def generate_exam_report_pdf(evaluation_data: Dict[str, Any], extracted_data: Optional[List[Dict[str, Any]]] = None) -> HttpResponse:
    """
    Generate a PDF report with enhanced formatting including question statements
    and comprehensive feedback.
    
    Args:
        evaluation_data: Dictionary containing evaluation results
        extracted_data: Optional list of extracted questions with statements and answers
        
    Returns:
        HttpResponse with PDF attachment
    """
    # Create a file-like buffer to receive PDF data
    buffer = io.BytesIO()
    
    # Create the PDF object, using the buffer as its "file"
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        rightMargin=72, 
        leftMargin=72, 
        topMargin=72, 
        bottomMargin=72
    )
    
    # Get the default sample styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = styles["Title"]
    
    heading_style = ParagraphStyle(
        name='HeadingStyle',
        parent=styles['Heading2'],
        alignment=1,  # Center alignment
        spaceAfter=12
    )
    
    normal_style = ParagraphStyle(
        name='NormalStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceBefore=6,
        spaceAfter=6
    )
    
    question_style = ParagraphStyle(
        name='QuestionStyle',
        parent=styles['Heading3'],
        fontSize=12,
        leading=16,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.navy
    )
    
    feedback_style = ParagraphStyle(
        name='FeedbackStyle',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        leftIndent=20,
        spaceBefore=6,
        spaceAfter=6
    )
    
    info_style = ParagraphStyle(
        name='InfoStyle',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        leftIndent=10,
        spaceBefore=3,
        spaceAfter=3,
        backColor=colors.lightgrey
    )
    
    # Extract data
    evaluations = evaluation_data.get("evaluations", [])
    
    # Calculate total score if not provided
    if "total_score" in evaluation_data:
        total_score = evaluation_data["total_score"]
    else:
        scores = [ev.get("score", ev.get("similarity_percentage", 0)) for ev in evaluations]
        total_score = round(sum(scores) / len(scores)) if scores else 0
    
    overall_feedback = evaluation_data.get("overall_feedback", "No overall feedback available.")
    
    # Create a mapping from question_no to extracted data
    extracted_map = {}
    if extracted_data:
        for item in extracted_data:
            question_no = item.get("question_no", "")
            if question_no:
                extracted_map[question_no] = item
    
    # Start building the document content
    content = []
    
    # Title
    content.append(Paragraph("Exam Grading Report", title_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Summary section with improved styling
    content.append(Paragraph("Summary", heading_style))
    
    summary_data = [
        ["Total Questions:", str(len(evaluations))],
        ["Overall Score:", f"{total_score}% ({total_score}/100)"]
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    
    content.append(summary_table)
    content.append(Spacer(1, 0.25*inch))
    
    # Question scores section with better formatting
    content.append(Paragraph("Question Details", heading_style))
    
    for evaluation in evaluations:
        question_no = evaluation.get("question_no", "")
        score = evaluation.get("score", evaluation.get("similarity_percentage", 0))
        feedback = evaluation.get("feedback", "No feedback available")
        
        # Get question statement and answer from either evaluation or extracted data
        question_statement = evaluation.get("question_statement", "")
        answer = evaluation.get("complete_answer", "")
        
        # If not in evaluation, try to get from extracted data
        if (not question_statement or not answer) and question_no in extracted_map:
            extracted = extracted_map[question_no]
            if not question_statement:
                question_statement = extracted.get("question_statement", "No question statement available")
            if not answer:
                answer = extracted.get("complete_answer", "No answer provided")
        
        # If still not found, use defaults
        if not question_statement:
            question_statement = "No question statement available"
        if not answer:
            answer = "No answer provided"
        
        # Clean text to remove any HTML tags
        question_statement = clean_html(question_statement)
        answer = clean_html(answer)
        feedback = clean_html(feedback)
        
        # Create a score label with color
        score_color = colors.green if score >= 80 else (colors.orange if score >= 60 else colors.red)
        
        # Question header with score
        question_header = f"Question {question_no}: {score}% ({score}/100)"
        content.append(Paragraph(question_header, question_style))
        
        # Question statement with better formatting
        content.append(Paragraph("<b>Statement:</b>", normal_style))
        content.append(Paragraph(question_statement, info_style))
        
        # Student answer with better formatting
        content.append(Paragraph("<b>Student Answer:</b>", normal_style))
        content.append(Paragraph(answer, info_style))
        
        # Feedback with color coding
        content.append(Paragraph("<b>Feedback:</b>", normal_style))
        feedback_para = Paragraph(feedback, feedback_style)
        content.append(feedback_para)
        
        content.append(Spacer(1, 0.25*inch))
    
    # Summary and recommendations section
    content.append(Paragraph("Summary and Recommendations", heading_style))
    
    # Split the overall feedback by lines for better formatting
    for line in overall_feedback.split("\n"):
        if line.strip():
            content.append(Paragraph(line, normal_style))
    
    # Build the PDF document
    doc.build(content)
    
    # Get the value of the BytesIO buffer
    pdf_value = buffer.getvalue()
    buffer.close()
    
    # Create the HTTP response with PDF content
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="exam_report.pdf"'
    response.write(pdf_value)
    
    return response

def download_report_pdf(request):
    """
    View function to generate and download the exam report PDF.
    Extracts evaluation data from session and generates PDF.
    """
    try:
        # Get evaluation data from session
        evaluation_json = request.session.get('evaluation_json')
        if not evaluation_json:
            return HttpResponse("No evaluation data found", status=404)
        
        # Parse the evaluation data
        if isinstance(evaluation_json, str):
            evaluation_data = json.loads(evaluation_json)
        else:
            evaluation_data = evaluation_json
            
        # Try to get extracted data from session as well
        extracted_data = None
        if 'extracted_data' in request.session:
            extracted_data = request.session.get('extracted_data')
            
        # Generate the PDF
        return generate_exam_report_pdf(evaluation_data, extracted_data)
    
    except Exception as e:
        import traceback
        error_msg = f"Error generating PDF report: {str(e)}\n{traceback.format_exc()}"
        return HttpResponse(error_msg, status=500, content_type='text/plain')