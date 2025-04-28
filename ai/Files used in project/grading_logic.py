import os
from typing import List, Dict
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import Document
from langchain_groq import ChatGroq
from langchain.retrievers import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# --- Define Pydantic Models for the Evaluation Report ---
class QuestionEvaluation(BaseModel):
    """Represents the evaluation metrics for a single question."""
    question_no: str = Field(description="The unique identifier for the question being evaluated.")
    similarity_percentage: int = Field(description="The estimated similarity percentage of the student's answer compared to a correct answer (0-100).")
    feedback: str = Field(description="Feedback explaining the similarity percentage, highlighting correct aspects, missing parts, or errors. This should be a concise explanation.")

class ExamEvaluationReport(BaseModel):
    """Represents the complete evaluation report for an exam."""
    evaluations: List[QuestionEvaluation] = Field(description="A list of evaluations for each extracted question.")


# --- Initialize Variables for Storing Context, Answers, and Questions ---
accumulated_context = ""  # Store the combined context of all previously evaluated questions
accumulated_answers: Dict[str, str] = {}  # Store the answers of all previously evaluated questions
accumulated_questions: Dict[str, str] = {}  # Store the questions of all previously evaluated questions

# --- Define a Mock Function to Retrieve Document (As an Example) ---
# In the real script, replace this with your actual retriever (e.g., Chroma, FAISS)
def retrieve_documents(query: str) -> List[Document]:
    """Mock function to simulate document retrieval for a question."""
    # Simulating retrieval by returning mock documents
    return [Document(page_content=f"Simulated context for query: {query}")]

# --- 2. Retrieve and Combine All Context ---
def get_combined_context() -> str:
    """Combine all context available for the exam."""
    combined_context = ""
    # Simulate document retrieval for all questions in the exam
    for question_data in extracted_exam_data_dict.get('extracted_questions', []):
        question_statement = question_data.get('question_statement', "")
        if question_statement:
            retrieved_docs = retrieve_documents(question_statement)
            combined_context += "\n\n".join([doc.page_content for doc in retrieved_docs])
    return combined_context


# --- 3. Evaluate Each Question with Accumulated Context, Answers, and Questions ---
evaluation_results: List[QuestionEvaluation] = []

for question_data in extracted_exam_data_dict.get('extracted_questions', []):
    question_no = question_data.get('question_no', 'Unknown')
    question_statement = question_data.get('question_statement', '')
    question_answer = question_data.get('complete_answer', '')

    if not question_statement or not question_answer:
        print(f"Skipping evaluation for Question {question_no} due to missing statement or answer.")
        continue

    print(f"Evaluating Question {question_no} with accumulated context...")

    # Retrieve context for the current question
    current_context = "No relevant context found."  # Default context if retrieval fails or is not available
    try:
        print(f"  Retrieving context for Question {question_no}...")
        retrieved_docs = retrieve_documents(question_statement)  # Retrieve context based on the question statement
        print(f"  Retrieved {len(retrieved_docs)} documents for question: {question_statement}")
        
        # Combine the content of the retrieved documents
        current_context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        if not current_context.strip():
            current_context = "No relevant context found for this question."
    except Exception as e:
        print(f"  An error occurred during context retrieval for Question {question_no}: {e}")
        current_context = "Error retrieving context."  # Indicate failure in context retrieval

    # Combine the current context with all previous contexts
    combined_context = accumulated_context + "\n\n---\n\n" + current_context

    # Store the current question's answer and question
    accumulated_answers[question_no] = question_answer
    accumulated_questions[question_no] = question_statement

    # Include previous answers and questions in the context
    context_with_answers_and_questions = combined_context + "\n\n---\n\nPrevious Questions and Answers:\n"
    for prev_q_no, prev_question in accumulated_questions.items():
        prev_answer = accumulated_answers.get(prev_q_no, "No answer provided.")
        context_with_answers_and_questions += f"Question {prev_q_no}: {prev_question}\nAnswer: {prev_answer}\n"

    # Simulating the evaluation (in the real script, replace this with actual model evaluation logic)
    try:
        # Assuming the model evaluates the question using the combined context
        evaluation = QuestionEvaluation(
            question_no=question_no,
            similarity_percentage=90,  # Dummy similarity value
            feedback=f"Good answer for Question {question_no}. More details required for clarity."
        )
        evaluation_results.append(evaluation)
        print(f"  Evaluation complete for Question {question_no}.")
        
        # Update the accumulated context and answers for the next question
        accumulated_context = combined_context  # Update context to include the current question's context

    except ValidationError as e:
        print(f"  Validation Error for Question {question_no}: {e}")
    except Exception as e:
        print(f"  An unexpected error occurred during evaluation for Question {question_no}: {e}")
        print("  Skipping evaluation for this question.")

# --- 4. Generate the Final Evaluation Report ---
final_report = ExamEvaluationReport(evaluations=evaluation_results)

# --- 5. Print the Final Report JSON ---
print("\n--- Final Exam Evaluation Report ---")
print(final_report.json(indent=4))  # Printing the final evaluation report as JSON
