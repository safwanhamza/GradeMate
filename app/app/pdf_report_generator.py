import io
from typing import List, Dict, Any, Optional
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import json

def generate_exam_report_pdf(evaluation_data: Dict[str, Any]) -> HttpResponse:
    """
    Generate a PDF report with enhanced formatting including question statements
    and comprehensive feedback.
    
    Args:
        evaluation_data: Dictionary containing evaluation results
        
    Returns:
        HttpResponse with PDF attachment
    """
    # Create a file-like buffer to receive PDF data
    buffer = io.BytesIO()
    
    # Create the PDF object, using the buffer as its "file"
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Get the default sample styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = styles["Title"]
    
    heading_style = styles["Heading2"]
    heading_style.alignment = 1  # Center alignment
    
    normal_style = styles["Normal"]
    normal_style.fontSize = 10
    normal_style.leading = 14
    
    question_style = ParagraphStyle(
        name='QuestionStyle',
        parent=styles['Heading3'],
        fontSize=12,
        leading=16
    )
    
    feedback_style = ParagraphStyle(
        name='FeedbackStyle',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        leftIndent=20
    )
    
    # Extract data
    evaluations = evaluation_data.get("evaluations", [])
    total_score = evaluation_data.get("total_score", 0)
    overall_feedback = evaluation_data.get("overall_feedback", "No overall feedback available.")
    
    # Start building the document content
    content = []
    
    # Title
    content.append(Paragraph("Exam Grading Report", title_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Summary section
    summary_data = [
        ["Total Questions:", str(len(evaluations))],
        ["Overall Score:", f"{total_score}% ({total_score}/100)"]
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    
    content.append(summary_table)
    content.append(Spacer(1, 0.25*inch))
    
    # Question scores section
    content.append(Paragraph("Question Scores:", heading_style))
    content.append(Spacer(1, 0.15*inch))
    
    for evaluation in evaluations:
        question_no = evaluation.get("question_no", "")
        question_text = evaluation.get("question_statement", "No question statement available")
        answer = evaluation.get("complete_answer", "No answer provided")
        score = evaluation.get("score", 0)
        feedback = evaluation.get("feedback", "No feedback available")
        
        # Question header with score
        question_header = f"Question {question_no}: {score}% ({score}/100)"
        content.append(Paragraph(question_header, question_style))
        
        # Question statement
        content.append(Paragraph(f"<b>Statement:</b> {question_text}", normal_style))
        
        # Student answer
        content.append(Paragraph(f"<b>Student Answer:</b> {answer}", normal_style))
        
        # Feedback
        content.append(Paragraph(f"<b>Feedback:</b> {feedback}", feedback_style))
        
        content.append(Spacer(1, 0.25*inch))
    
    # Summary and recommendations section
    content.append(Paragraph("Summary and Recommendations:", heading_style))
    content.append(Spacer(1, 0.15*inch))
    
    # Split the overall feedback by lines for better formatting
    for line in overall_feedback.split("\n"):
        if line.strip():
            content.append(Paragraph(line, normal_style))
            content.append(Spacer(1, 0.1*inch))
    
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