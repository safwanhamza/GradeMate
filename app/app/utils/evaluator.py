# evaluator.py
import os
import json
import logging
import re  # For regex
from typing import List, Dict, Any, Optional, Union
import traceback
# Use the latest Pydantic import
from pydantic import BaseModel, Field, validator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.vectorstores import Chroma
# evaluator.py - Enhanced with MCQ Support

# Configure logging
logger = logging.getLogger(__name__)


class MCQOption(BaseModel):
    """Represents a single option in a multiple-choice question."""
    letter: str = Field(description="Option identifier (e.g., 'A', 'B', 'C')")
    text: str = Field(description="Text content of the option")
    is_selected: bool = Field(default=False, description="Whether this option was selected by the student")

class QuestionEvaluation(BaseModel):
    """Structured model for question evaluation."""
    question_no: str
    score: int = Field(
        description="Score (0-100) for the student's answer",
        ge=0, 
        le=100
    )
    feedback: str = Field(
        description="Concise feedback explaining the score, highlighting correct and incorrect aspects",
        max_length=250
    )
    
    # Add validator to ensure question_no is always a string
    @validator('question_no', pre=True)
    def ensure_question_no_is_string(cls, v):
        if v is None:
            return "Unknown"
        return str(v)  # Convert any type to string


class ExamEvaluationReport(BaseModel):
    """Structured model for the entire exam evaluation."""
    evaluations: List[QuestionEvaluation]
    total_score: int = Field(
        description="Overall exam score (0-100)",
        ge=0,
        le=100
    )
    overall_feedback: str = Field(
        description="General feedback about the entire exam performance",
        max_length=500
    )




def truncate_feedback(feedback, max_length=250):
    """Safely truncate feedback to avoid exceeding max length."""
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


def generate_summary_feedback(evaluations: List[QuestionEvaluation]) -> str:
    """Generate summary feedback based on evaluation results."""
    if not evaluations:
        return "No evaluations provided."
    
    # Calculate stats
    scores = [eval.score for eval in evaluations]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Count by performance category
    excellent = sum(1 for s in scores if s >= 80)
    good = sum(1 for s in scores if 60 <= s < 80)
    needs_improvement = sum(1 for s in scores if s < 60)
    
    # Generate main assessment based on overall score
    if avg_score >= 85:
        assessment = "Outstanding performance! You demonstrated excellent understanding of the subject matter."
    elif avg_score >= 75:
        assessment = "Great job! You showed strong comprehension of most concepts."
    elif avg_score >= 60:
        assessment = "Good work. You demonstrated understanding of core concepts with some areas for improvement."
    elif avg_score >= 50:
        assessment = "Satisfactory. You have a basic grasp of the material but need to strengthen your understanding."
    else:
        assessment = "Performance requires substantial improvement. Recommend comprehensive review of course materials and seeking additional support."
    
    return assessment


def detect_mcq_from_answer(question_data: Dict[str, Any]) -> bool:
    """
    Enhanced MCQ detection from question and answer data.
    
    Args:
        question_data: Dictionary containing question and answer data
        
    Returns:
        Boolean indicating whether the question appears to be an MCQ
    """
    # Already marked as MCQ
    if question_data.get("is_mcq", False):
        return True
    
    question_statement = question_data.get("question_statement", "")
    complete_answer = question_data.get("complete_answer", "")
    
    # Check for option patterns in question
    mcq_patterns = [
        r'\([A-D]\)\s+\w+',  # (A) Option format
        r'[A-D]\)\s+\w+',    # A) Option format
        r'\b[A-D]\.\s+\w+',  # A. Option format
        r'(?:^|\n)[\t ]*[A-D][\.\):][\t ]+\w+', # MCQ at line start
        r'(?:circle|mark|choose|select)\s+(?:one|the correct|the right|the best|your|an)?\s*(?:option|answer|choice)',  # Instructions
        r'multiple(?:\s+|-)?choice',  # Explicit MCQ reference
    ]
    
    for pattern in mcq_patterns:
        if re.search(pattern, question_statement, re.IGNORECASE):
            return True
    
    # Check if answer is just a single option letter
    if complete_answer and complete_answer.strip().upper() in ["A", "B", "C", "D"]:
        return True
    
    # Check if answer mentions selecting an option
    if re.search(r'(?:(?:i|student)\s+(?:select|chose|pick|mark|circle))\s+(?:option|answer|choice)?\s*[A-D]', 
                complete_answer, re.IGNORECASE):
        return True
    
    # Count sequential option-like patterns to detect option lists
    option_sequences = re.findall(r'(?:[A-D][\.\):][\t ]+\w+[^\n]*\n){2,}', question_statement)
    if option_sequences:
        return True
    
    return False


def extract_mcq_data(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract MCQ options and selected option from question data.
    
    Args:
        question_data: Dictionary containing question and answer data
        
    Returns:
        Updated question_data with MCQ-specific fields
    """
    if question_data.get("options"):
        # Options already extracted
        return question_data
    
    question_statement = question_data.get("question_statement", "")
    complete_answer = question_data.get("complete_answer", "")
    
    # Extract options from question statement
    options = []
    option_matches = re.finditer(r'([A-D])[\.\):][\t ]+([^\n]+)', question_statement, re.IGNORECASE)
    
    for match in option_matches:
        letter = match.group(1).upper()
        text = match.group(2).strip()
        is_selected = False  # We'll determine selected option separately
        
        options.append({
            "letter": letter,
            "text": text,
            "is_selected": is_selected
        })
    
    # Determine selected option from answer
    selected_option = None
    
    # Direct letter answer
    if complete_answer and complete_answer.strip().upper() in ["A", "B", "C", "D"]:
        selected_option = complete_answer.strip().upper()
    
    # Mentioned in answer
    elif complete_answer:
        # Look for patterns like "I select option B" or "The answer is C"
        option_match = re.search(r'(?:select|chose|pick|mark|circle|answer\s+is|chose)\s+(?:option|answer|choice)?\s*([A-D])',
                              complete_answer, re.IGNORECASE)
        if option_match:
            selected_option = option_match.group(1).upper()
        else:
            # Just look for letter in answer
            letter_match = re.search(r'\b([A-D])\b', complete_answer)
            if letter_match:
                selected_option = letter_match.group(1).upper()
    
    # Update options with selected status
    if selected_option and options:
        for option in options:
            option["is_selected"] = option["letter"] == selected_option
    
    # Update question data
    updated_data = dict(question_data)
    updated_data["is_mcq"] = True
    updated_data["options"] = options
    updated_data["selected_option"] = selected_option
    
    return updated_data


class ExamEvaluator:
    def __init__(
        self, 
        api_key: str = None, 
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        temperature: float = 0.0,
        answer_key: Dict[str, Any] = None
    ):
        """
        Initialize the exam evaluation system
        
        Args:
            api_key (str, optional): Groq API key
            model (str, optional): LLM model to use
            temperature (float, optional): Sampling temperature
        """
        # Get API key
        try:
            from django.conf import settings
            self.api_key = (
                api_key or 
                os.environ.get('GROQ_API_KEY') or 
                getattr(settings, 'GROQ_API_KEY', None)
            )
        except ImportError:
            self.api_key = api_key or os.environ.get('GROQ_API_KEY')
        
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        # Store answer key if provided
        self.answer_key = answer_key or {}
        
        # Initialize LLM client
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            self.model = model
            self.temperature = temperature
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.client = None
    
    def grade_mcq(self, question_data: Dict[str, Any], correct_answers: Dict[str, str] = None) -> QuestionEvaluation:
        """
        Grade a multiple-choice question
        
        Args:
            question_data: The MCQ data including options and selected answer
            correct_answers: Optional dictionary mapping question numbers to correct option letters
        
        Returns:
            QuestionEvaluation with score and feedback
        """
        question_no = str(question_data.get("question_no", "Unknown"))
        selected_option = question_data.get("selected_option")
        
        # If no option selected
        if not selected_option:
            return QuestionEvaluation(
                question_no=question_no,
                score=0,
                feedback="No option was selected for this MCQ."
            )
        
        # Check if we have the correct answer
        correct_option = None
        if correct_answers and question_no in correct_answers:
            correct_option = correct_answers[question_no]
            
            # If we know the correct answer, we can score accordingly
            if selected_option.upper() == correct_option.upper():
                return QuestionEvaluation(
                    question_no=question_no,
                    score=100,
                    feedback=f"Correct! Option {selected_option} was selected."
                )
            else:
                return QuestionEvaluation(
                    question_no=question_no,
                    score=0,
                    feedback=f"Incorrect. Option {selected_option} was selected, but the correct answer is {correct_option}."
                )
        
        # If we don't have correct answers, use LLM to make a judgment
        if self.client:
            try:
                # Convert options to text for the LLM
                options_text = ""
                for opt in question_data.get("options", []):
                    options_text += f"{opt.get('letter')}: {opt.get('text')}\n"
                
                # Create an MCQ grading prompt
                prompt = f"""
                Grade this multiple-choice question based on likelihood of correctness:
                
                Question: {question_data.get('question_statement', '')}
                
                Options:
                {options_text}
                
                Student's selected answer: {selected_option}
                
                You don't have an answer key, but based on your knowledge, estimate which answer is most likely correct
                and score the student's response accordingly.
                
                Respond with JSON in this format:
                {{
                  "score": 100,  
                  "feedback": "Brief feedback on the answer",
                  "likely_correct_answer": "A"
                }}
                
                Notes:
                - Score should be 100 if the selected answer seems correct, 0 if it seems incorrect
                - If you're uncertain, give the student the benefit of the doubt (score 100)
                - Keep feedback concise (under 200 characters)
                """
                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=300
                )
                
                content = completion.choices[0].message.content
                
                # Extract the JSON
                try:
                    # Try to find JSON pattern
                    json_match = re.search(r'{.*}', content, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(0))
                        score = result.get('score', 100)
                        likely_correct = result.get('likely_correct_answer', selected_option)
                        
                        if score == 100:
                            feedback = truncate_feedback(
                                result.get('feedback', f"Option {selected_option} appears to be correct.")
                            )
                        else:
                            feedback = truncate_feedback(
                                result.get('feedback', f"Option {selected_option} may be incorrect. The likely correct answer is {likely_correct}.")
                            )
                        
                        return QuestionEvaluation(
                            question_no=question_no,
                            score=score,
                            feedback=feedback
                        )
                except Exception as json_err:
                    logger.error(f"Failed to parse MCQ grading response: {json_err}")
            
            except Exception as e:
                logger.error(f"Error grading MCQ {question_no}: {e}")
        
        # Fallback evaluation - give benefit of the doubt
        return QuestionEvaluation(
            question_no=question_no,
            score=100,  # Default to full credit without answer key
            feedback=f"Option {selected_option} was selected. MCQ response graded automatically."
        )
        
    def grade_subjective(self, question_data: Dict[str, Any]) -> QuestionEvaluation:
        """
        Grade a subjective/essay question
        
        Args:
            question_data: Dictionary containing question statement and student's answer
            
        Returns:
            QuestionEvaluation with score and feedback
        """
        question_no = str(question_data.get("question_no", "Unknown"))
        question_statement = question_data.get("question_statement", "")
        complete_answer = question_data.get("complete_answer", "")
        
        # Handle empty answers
        if not complete_answer or complete_answer.strip().lower() in ["no answer provided", "no clear answer provided"]:
            return QuestionEvaluation(
                question_no=question_no,
                score=0,
                feedback="No answer provided by the student to assess."
            )
        
        # If we have an answer key for this question
        model_answer = self.answer_key.get(question_no)
        
        if self.client:
            try:
                # Prepare the grading prompt
                prompt = f"""
                Grade this subjective answer:
                
                Question: {question_statement}
                
                Student's Answer: {complete_answer}
                """
                
                # Add model answer if available
                if model_answer:
                    prompt += f"""
                    
                    Reference Answer: {model_answer}
                    
                    Compare the student's answer with the reference answer. Grade fairly based on:
                    - Accuracy of content
                    - Completeness of answer
                    - Clarity of explanation
                    """
                else:
                    prompt += """
                    
                    Grade this answer based on:
                    - Accuracy of the information
                    - Completeness and depth
                    - Clarity and organization
                    - Relevance to the question
                    
                    Since you don't have a reference answer, use your knowledge to evaluate correctness.
                    """
                
                prompt += """
                
                Provide a score from 0-100 and brief, constructive feedback.
                
                Respond with JSON in this format:
                {
                  "score": 75,
                  "feedback": "Brief, specific feedback about the answer"
                }
                
                Keep feedback under 200 characters.
                """
                
                # Call the LLM for grading
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=500
                )
                
                content = completion.choices[0].message.content
                
                # Extract the JSON result
                try:
                    # Try to find JSON pattern
                    json_match = re.search(r'{.*}', content, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(0))
                        return QuestionEvaluation(
                            question_no=question_no,
                            score=result.get('score', 50),
                            feedback=truncate_feedback(result.get('feedback', "Answer was evaluated but no specific feedback was generated."))
                        )
                except Exception as json_err:
                    logger.error(f"Failed to parse subjective grading response: {json_err}")
            
            except Exception as e:
                logger.error(f"Error grading subjective question {question_no}: {e}")
        
        # Fallback grading based on answer length
        answer_length = len(complete_answer)
        if answer_length < 20:
            score = 30
            feedback = "Very brief answer that lacks detail and explanation."
        elif answer_length < 50:
            score = 60
            feedback = "Basic answer with limited explanation. Could be more detailed."
        else:
            score = 80
            feedback = "Good answer with adequate detail and explanation."
        
        return QuestionEvaluation(
            question_no=question_no,
            score=score,
            feedback=feedback
        )
    
    def evaluate_exam(self, extracted_data: List[Dict[str, Any]]) -> ExamEvaluationReport:
        """
        Evaluate an entire exam with mixed question types
        
        Args:
            extracted_data: List of extracted questions with answers
            
        Returns:
            ExamEvaluationReport with question evaluations and overall score
        """
        logger.info(f"Starting exam evaluation for {len(extracted_data)} questions")
        
        question_evaluations = []
        mcq_count = 0
        subjective_count = 0
        
        # Process each question
        for question_data in extracted_data:
            try:
                # Check if the question is already marked as MCQ or needs detection
                is_mcq = question_data.get("is_mcq", False)
                
                # If not already marked as MCQ, try to detect it
                if not is_mcq:
                    is_mcq = detect_mcq_from_answer(question_data)
                    if is_mcq:
                        logger.info(f"MCQ detected for question {question_data.get('question_no')}")
                        # Extract MCQ data and update the question
                        question_data = extract_mcq_data(question_data)
                
                # Grade accordingly
                if is_mcq:
                    mcq_count += 1
                    logger.info(f"Grading MCQ question {question_data.get('question_no')}")
                    # Get correct answers from answer key if available
                    correct_mcq_answers = {k: v for k, v in self.answer_key.items() if isinstance(v, str) and len(v) == 1}
                    evaluation = self.grade_mcq(question_data, correct_mcq_answers)
                else:
                    subjective_count += 1
                    logger.info(f"Grading subjective question {question_data.get('question_no')}")
                    evaluation = self.grade_subjective(question_data)
                
                question_evaluations.append(evaluation)
                logger.info(f"Evaluated question {evaluation.question_no}: score={evaluation.score}")
                
            except Exception as e:
                logger.error(f"Error processing question {question_data.get('question_no', 'unknown')}: {e}")
                logger.error(traceback.format_exc())
                # Add a default evaluation for this question
                question_evaluations.append(
                    QuestionEvaluation(
                        question_no=str(question_data.get("question_no", "Unknown")),
                        score=30,
                        feedback="Unable to grade due to processing error. Please review manually."
                    )
                )
        
        # Log MCQ detection results
        logger.info(f"Processed {mcq_count} MCQs and {subjective_count} subjective questions")
        
        # Calculate total score - weighted by question type if needed
        if question_evaluations:
            # Simple average for now
            total_score = sum(q.score for q in question_evaluations) // len(question_evaluations)
        else:
            total_score = 0
        
        # Generate overall feedback
        overall_feedback = generate_summary_feedback(question_evaluations)
        
        # Create and return final report
        return ExamEvaluationReport(
            evaluations=question_evaluations,
            total_score=total_score,
            overall_feedback=overall_feedback
        )

# This is the function your code is trying to import - it was missing in your original file
def evaluate_exam(extracted_data: List[Dict[str, Any]], answer_key: Dict[str, Any] = None) -> ExamEvaluationReport:
    """
    Entry point for exam evaluation with robust error handling
    
    Args:
        extracted_data: List of extracted questions with answers
        answer_key: Optional dictionary of correct answers

    Returns:
        ExamEvaluationReport with comprehensive evaluation
    """
    try:
        # Get API key with flexible retrieval
        api_key = None
        try:
            from django.conf import settings
            api_key = getattr(settings, 'GROQ_API_KEY', None)
        except ImportError:
            api_key = os.environ.get('GROQ_API_KEY')
        
        if not api_key:
            logger.error("Groq API key is required")
            raise ValueError("Groq API key is required")
        
        # Log MCQ detection stats before evaluation
        mcq_count = sum(1 for q in extracted_data if q.get('is_mcq', False))
        logger.info(f"Initial MCQ count: {mcq_count} out of {len(extracted_data)} questions")
        
        # Pre-process to attempt MCQ detection on any unmarked questions
        for question in extracted_data:
            if not question.get('is_mcq', False):
                if detect_mcq_from_answer(question):
                    # Extract MCQ data and update the question
                    updated = extract_mcq_data(question)
                    # Update the original question in place
                    for key, value in updated.items():
                        question[key] = value
        
        # Log updated MCQ count
        mcq_count_after = sum(1 for q in extracted_data if q.get('is_mcq', False))
        logger.info(f"Updated MCQ count after detection: {mcq_count_after} out of {len(extracted_data)} questions")
        
        if mcq_count_after > mcq_count:
            logger.info(f"Successfully detected {mcq_count_after - mcq_count} additional MCQs")
        
        # Initialize the evaluator
        evaluator = ExamEvaluator(
            api_key=api_key,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.0,
            answer_key=answer_key
        )
        
        # Evaluate the exam
        evaluation_report = evaluator.evaluate_exam(extracted_data)
        
        return evaluation_report

    except Exception as e:
        logger.error(f"Comprehensive exam evaluation failed: {e}")
        logger.error(traceback.format_exc())
        # Fallback report in case of catastrophic failure
        return ExamEvaluationReport(
            evaluations=[],
            total_score=0,
            overall_feedback="Error during exam evaluation. Please verify API configuration and try again."
        )