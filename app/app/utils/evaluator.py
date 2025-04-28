# evaluator.py

import os
import json
import logging
import re  # Add re module for regex
from typing import List, Dict, Any, Tuple

# Use the latest Pydantic import
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser  # Remove OutputParsingException
from langchain_groq import ChatGroq
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logger = logging.getLogger(__name__)

class QuestionEvaluation(BaseModel):
    """Structured model for question evaluation."""
    question_no: str
    score: int = Field(
        description="Similarity percentage (0-100) of the student's answer to the correct answer",
        ge=0, 
        le=100
    )
    feedback: str = Field(
        description="Concise feedback explaining the similarity score, highlighting correct and incorrect aspects",
        max_length=500
    )


class ExamEvaluationReport(BaseModel):
    """Structured model for the entire exam evaluation."""
    evaluations: List[QuestionEvaluation]
    total_score: int = Field(
        description="Overall exam score calculated from individual question scores",
        ge=0,
        le=100
    )
    overall_feedback: str = Field(
        description="General feedback about the entire exam performance",
        max_length=500
    )

def truncate_feedback(feedback, max_length=500):
    """Safely truncate feedback to the specified maximum length."""
    if not feedback or len(feedback) <= max_length:
        return feedback
    
    # Try to truncate at sentence or punctuation
    truncation_points = ['. ', '! ', '? ', '; ']
    for point in truncation_points:
        last_point = feedback[:max_length].rfind(point)
        if last_point > max_length // 2:  # Make sure we don't truncate too early
            return feedback[:last_point+1]
    
    # If no good truncation point, just truncate and add ellipsis
    return feedback[:max_length-3] + "..."

class LangchainExamEvaluator:
    def __init__(
        self, 
        api_key: str = None, 
        model: str = "deepseek-r1-distill-llama-70b",
        temperature: float = 0.4,
        embedding_model_path: str = "./chroma_db"
    ):
        """
        Initialize the exam evaluation system
        
        Args:
            api_key (str, optional): Groq API key
            model (str, optional): LLM model to use
            temperature (float, optional): Sampling temperature
            embedding_model_path (str, optional): Path to Chroma vector store
        """
        # Comprehensive API key retrieval
        from django.conf import settings
        self.api_key = (
            api_key or 
            os.environ.get('GROQ_API_KEY') or 
            getattr(settings, 'GROQ_API_KEY', None)
        )
        
        if not self.api_key:
            raise ValueError("Groq API key is required. Set it in environment variables or Django settings.")
        
        # Initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=QuestionEvaluation)
        
        # Create prompt template
        self.evaluation_prompt_template = PromptTemplate(
            template="""You are an expert academic grader tasked with evaluating a student's exam answer.

Evaluate the student's answer based on the question and the answer's comprehensiveness.

Question Number: {question_no}
Question: {question_statement}
Student's Answer: {complete_answer}

Detailed Evaluation Instructions:
1. Carefully assess the answer's correctness, completeness, and approach
2. Score the answer on a scale of 0-100:
   - 80-100: Very good answer with minor improvements possible
   - 70-79: Good answer with some key elements missing
   - 60-69: Satisfactory answer with significant gaps
   - 50-59: Partial understanding of the concept
   - 30-49: Minimal correct information
   - 0-29: Incorrect or completely irrelevant answer
3. Provide concise, constructive feedback (250 characters or less)

Your evaluation must be in valid JSON format as follows:
```json
{{
  "question_no": "{question_no}",
  "score": 75,
  "feedback": "Brief feedback about the answer here"
}}
Your evaluation in the format specified above:""",
input_variables=[
"question_no",
"question_statement",
"complete_answer"
]
        )
        
        # Initialize LLM
        self.llm = ChatGroq(
            temperature=temperature, 
            model_name=model, 
            api_key=self.api_key
        )
        
        # Setup vector store (optional)
        try:
            self.vector_store = Chroma(
                persist_directory=embedding_model_path,
                embedding_function=NomicEmbeddings(model="nomic-embed-text-v1")
            )
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
            self.vector_store = None



    def grade_question(self, question_no, question_statement, complete_answer) -> QuestionEvaluation:
        try:
            # Handle "No answer provided" case
            if not complete_answer or complete_answer.strip().lower() in ["no answer provided", "no clear answer provided"]:
                return QuestionEvaluation(
                    question_no=question_no,
                    score=0,
                    feedback="No answer was provided for this question."
                )
            
            # Modify the prompt to more explicitly require JSON format
            prompt_template = """You are an expert academic grader. Evaluate this answer:

    Question: {question_statement}
    Student's Answer: {complete_answer}

    Score the answer on a scale of 0-100:
    - 90-100: Exceptional understanding
    - 80-89: Very good
    - 70-79: Good with some gaps
    - 60-69: Satisfactory
    - 30-59: Minimal understanding
    - 0-29: Incorrect or irrelevant

    Provide ONLY valid JSON with this format, nothing else:
    {{
    "question_no": "{question_no}",
    "score": 75,
    "feedback": "Brief feedback here"
    }}

    IMPORTANT: No <think> tags, no additional text, JUST JSON."""

            # Get raw response from LLM
            response = self.llm.invoke(
                prompt_template.format(
                    question_no=question_no,
                    question_statement=question_statement,
                    complete_answer=complete_answer
                )
            )
            
            # Extract content from response
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up content by removing think tags and other artifacts
            cleaned_content = re.sub(r'</?think>|```json|```', '', content)
            cleaned_content = re.sub(r'^\s*\n', '', cleaned_content, flags=re.MULTILINE)
            
            # Try to extract JSON from the response using multiple approaches
            try:
                # First try: find JSON object pattern
                json_pattern = r'({[\s\S]*?})'
                json_match = re.search(json_pattern, cleaned_content)
                
                if json_match:
                    # Extract and parse the matched JSON
                    json_str = json_match.group(1)
                    result_dict = json.loads(json_str)
                    
                    # Create QuestionEvaluation from the parsed dict
                    result = QuestionEvaluation(
                        question_no=result_dict.get('question_no', question_no),
                        score=result_dict.get('score', 30),
                        feedback=truncate_feedback(result_dict.get('feedback', "No specific feedback provided."), 500)
                    )
                    return result
                
                # If no JSON found, try to extract structured information
                logger.error(f"No valid JSON found. Trying to extract structured info...")
                
                # Extract score using regex
                score_match = re.search(r'(?:score|points?|grade|marks?)(?:\s*:?\s*)(\d+)', cleaned_content, re.IGNORECASE)
                score = int(score_match.group(1)) if score_match else 30
                
                # Extract feedback - look for keywords
                feedback_text = "Unable to extract structured feedback."
                for keyword in ["feedback", "comment", "assessment", "evaluation"]:
                    if keyword in cleaned_content.lower():
                        parts = cleaned_content.lower().split(keyword)
                        if len(parts) > 1:
                            # Take text after the keyword
                            raw_feedback = parts[1].strip()
                            # Take first 200 chars or up to next heading
                            end_idx = min(200, len(raw_feedback))
                            heading_match = re.search(r'^\s*[A-Z][A-Za-z\s]+:', raw_feedback)
                            if heading_match:
                                end_idx = min(end_idx, heading_match.start())
                            feedback_text = raw_feedback[:end_idx].strip()
                            break
                
                # If we have no feedback yet, just take the most reasonable text chunk
                if feedback_text == "Unable to extract structured feedback.":
                    # Remove very long lines (likely code) and take a few sentences
                    lines = [l for l in cleaned_content.split('\n') if len(l) < 100]
                    readable_text = ' '.join(lines)
                    sentences = re.split(r'(?<=[.!?])\s+', readable_text)
                    feedback_text = ' '.join(sentences[:3])[:200].strip()
                
                return QuestionEvaluation(
                    question_no=question_no,
                    score=score,
                    feedback=feedback_text
                )
                    
            except Exception as json_err:
                logger.error(f"Error parsing LLM response: {json_err}")
                logger.error(f"Raw response: {content[:500]}...")
                
                # Create a fallback evaluation with information extracted from text
                # Look for numeric values that might represent scores
                score = 30  # Default score
                
                # Try to extract a score from the text
                score_patterns = [
                    r'score[^\d]*(\d+)',
                    r'(\d+)[^\d]*points',
                    r'grade[^\d]*(\d+)',
                    r'(\d+)[^\d]*percent',
                    r'(\d+)[^\d]*%',
                    r'(\d+)[^\d]*/[^\d]*100'
                ]
                
                for pattern in score_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        try:
                            score = int(match.group(1))
                            if 0 <= score <= 100:  # Validate score range
                                break
                        except ValueError:
                            continue
                
                # Try to extract meaningful feedback
                # Just take the first couple of sentences that aren't obviously system text
                sentences = re.split(r'(?<=[.!?])\s+', content)
                filtered_sentences = [s for s in sentences if len(s) > 15 and "<" not in s and "```" not in s]
                feedback = " ".join(filtered_sentences[:2])[:200] if filtered_sentences else "Processing error occurred."
                
                return QuestionEvaluation(
                    question_no=question_no,
                    score=score,
                    feedback=feedback
                )
            
        except Exception as e:
            logger.error(f"Error grading question {question_no}: {e}")
            # Fallback evaluation
            return QuestionEvaluation(
                question_no=question_no,
                score=30,
                feedback="Unable to grade due to processing error."
            )




    def evaluate_exam(self, extracted_data: List[Dict[str, Any]]) -> ExamEvaluationReport:
        """
        Evaluate an entire exam
        
        Args:
            extracted_data (List[Dict]): Extracted exam questions and answers
        
        Returns:
            ExamEvaluationReport: Comprehensive exam evaluation
        """
        logger.info(f"Starting exam evaluation for {len(extracted_data)} questions")
        
        question_evaluations = []
        total_score = 0
        
        for question_data in extracted_data:
            try:
                # Extract necessary fields with robust handling
                question_no = question_data.get("question_no", "Unknown")
                question_statement = question_data.get("question_statement", "")
                complete_answer = question_data.get("complete_answer", "No answer provided")
                
                # Skip if crucial data is missing
                if not question_statement:
                    logger.warning(f"Skipping evaluation for question {question_no} due to missing question statement")
                    continue
                
                # Grade the question
                question_eval = self.grade_question(
                    question_no, 
                    question_statement, 
                    complete_answer
                )
                
                question_evaluations.append(question_eval)
                total_score += question_eval.score
            
            except Exception as e:
                logger.error(f"Error processing question {question_no}: {e}")
                # Add a default evaluation if processing fails
                question_evaluations.append(
                    QuestionEvaluation(
                        question_no=question_no,
                        score=30,
                        feedback="Unable to grade due to processing error."
                    )
                )
        
        # Calculate overall exam score and feedback
        if question_evaluations:
            total_score = total_score // len(question_evaluations)
        else:
            total_score = 0
        
        # Generate overall feedback based on total score
        if total_score >= 90:
            overall_feedback = "Outstanding performance! Demonstrates exceptional understanding and mastery of the subject matter."
        elif total_score >= 80:
            overall_feedback = "Excellent exam performance. Shows strong comprehension of key concepts with minimal errors."
        elif total_score >= 70:
            overall_feedback = "Good performance. Demonstrates solid understanding with room for improvement in some areas."
        elif total_score >= 60:
            overall_feedback = "Satisfactory performance. Basic understanding achieved, but significant improvement needed."
        else:
            overall_feedback = "Performance requires substantial improvement. Recommend comprehensive review of course materials and seeking additional support."
        
        # Create and return the comprehensive exam report
        exam_report = ExamEvaluationReport(
            evaluations=question_evaluations,
            total_score=total_score,
            overall_feedback=overall_feedback
        )
        
        logger.info(f"Completed exam evaluation. Total Score: {total_score}")
        return exam_report

def evaluate_exam(extracted_data: List[Dict[str, Any]]) -> ExamEvaluationReport:
    """
    Entry point for exam evaluation with robust error handling
    Args:
        extracted_data (List[Dict]): Extracted exam questions and answers

    Returns:
        ExamEvaluationReport: Comprehensive exam evaluation
    """
    try:
        # Initialize the evaluator with more flexible key retrieval
        evaluator = LangchainExamEvaluator()
        
        # Evaluate the exam
        evaluation_report = evaluator.evaluate_exam(extracted_data)
        
        return evaluation_report

    except Exception as e:
        logger.error(f"Comprehensive exam evaluation failed: {e}")
        # Fallback report in case of catastrophic failure
        return ExamEvaluationReport(
            evaluations=[],
            total_score=0,
            overall_feedback="Error during exam evaluation. Please verify API configuration and try again."
        )
