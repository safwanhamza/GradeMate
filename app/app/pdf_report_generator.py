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
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    heading_style = ParagraphStyle(name='HeadingStyle', parent=styles['Heading2'], alignment=1, spaceAfter=12)
    normal_style = ParagraphStyle(name='NormalStyle', parent=styles['Normal'], fontSize=10, leading=14, spaceBefore=6, spaceAfter=6)
    question_style = ParagraphStyle(name='QuestionStyle', parent=styles['Heading3'], fontSize=12, leading=16, spaceBefore=12, spaceAfter=6, textColor=colors.navy)
    feedback_style = ParagraphStyle(name='FeedbackStyle', parent=styles['BodyText'], fontSize=10, leading=14, leftIndent=20, spaceBefore=6, spaceAfter=6)
    info_style = ParagraphStyle(name='InfoStyle', parent=styles['BodyText'], fontSize=10, leading=14, leftIndent=10, spaceBefore=3, spaceAfter=3, backColor=colors.lightgrey)

    evaluations = evaluation_data.get("evaluations", [])
    total_score = evaluation_data.get("total_score", round(sum(ev.get("score", ev.get("similarity_percentage", 0)) for ev in evaluations) / len(evaluations)) if evaluations else 0)
    overall_feedback = evaluation_data.get("overall_feedback", "No overall feedback available.")

    extracted_map = {str(item.get("question_no")): item for item in (extracted_data or []) if item.get("question_no")}

    content = [
        Paragraph("Exam Grading Report", title_style),
        Spacer(1, 0.25 * inch),
        Paragraph("Summary", heading_style)
    ]

    summary_table = Table([
        ["Total Questions:", str(len(evaluations))],
        ["Overall Score:", f"{total_score}% ({total_score}/100)"]
    ], colWidths=[2 * inch, 3 * inch])

    summary_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))

    content.append(summary_table)
    content.append(Spacer(1, 0.25 * inch))
    content.append(Paragraph("Question Details", heading_style))

    for evaluation in evaluations:
        question_no = evaluation.get("question_no", "")
        score = evaluation.get("score", evaluation.get("similarity_percentage", 0))
        feedback = clean_html(evaluation.get("feedback", "No feedback available"))
        question_statement = evaluation.get("question_statement", "")
        answer = evaluation.get("complete_answer", "")

        qid = str(question_no)
        if (not question_statement or not answer) and qid in extracted_map:
            extracted = extracted_map[qid]
            question_statement = question_statement or extracted.get("question_statement", "No question statement available")

            if extracted.get("is_mcq"):
                options = extracted.get("options", [])
                selected_option = extracted.get("selected_option", "")
                option_lines = []

                for opt in options:
                    if isinstance(opt, dict):
                        # letter = opt.get("letter", "?")
                        opt_letter = opt.get("letter", "?")

                        text = opt.get("text", "")
                        selected = "(Selected)" if opt.get("is_selected") else ""
                    elif isinstance(opt, str):
                        import re
                        match = re.match(r'([A-D])[\.\)]\s*(.*)', opt.strip())
                        if match:
                            # letter, text = match.group(1), match.group(2)
                            opt_letter, text = match.group(1), match.group(2)

                        else:
                            # letter, text = "?", opt
                            opt_letter, text = "?", opt

                        selected = "(Selected)" if opt.strip().startswith(selected_option.strip()) else ""
                    else:
                        # letter, text, selected = "?", str(opt), ""
                        opt_letter, text, selected = "?", str(opt), ""


                    # option_lines.append(f"{letter}) {text} {selected}")
                    option_lines.append(f"{opt_letter}) {text} {selected}")


                answer = f"Selected Option: {selected_option or 'None'}\nChoices:\n" + "\n".join(option_lines)
            else:
                answer = answer or extracted.get("complete_answer", "No answer provided")

        question_statement = clean_html(question_statement or "No question statement available")
        answer = clean_html(answer or "No answer provided")

        question_header = f"Question {question_no}: {score}% ({score}/100)"
        content.extend([
            Paragraph(question_header, question_style),
            Paragraph("<b>Statement:</b>", normal_style),
            Paragraph(question_statement, info_style),
            Paragraph("<b>Student Answer:</b>", normal_style),
            Paragraph(answer, info_style),
            Paragraph("<b>Feedback:</b>", normal_style),
            Paragraph(feedback, feedback_style),
            Spacer(1, 0.25 * inch)
        ])

    content.append(Paragraph("Summary and Recommendations", heading_style))
    for line in overall_feedback.split("\n"):
        if line.strip():
            content.append(Paragraph(line, normal_style))

    doc.build(content)
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="exam_report.pdf"'
    response.write(buffer.getvalue())
    buffer.close()
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