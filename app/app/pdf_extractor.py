# pdf_extractor.py with enhanced MCQ detection
import os
import base64
import fitz  # PyMuPDF
from io import BytesIO
from typing import List, Tuple, Dict, Any
import json
import re
import logging
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f"pdf_extractor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define Pydantic models for structured output
class ExtractedQuestion(BaseModel):
    """Represents a single question and its answer."""
    question_no: Union[str, int] = Field(description="Question number or identifier")
    question_statement: str = Field(description="Complete question text")
    complete_answer: str = Field(description="Student's answer or 'No answer provided'")
    is_mcq: bool = Field(default=False, description="Whether this question is a multiple-choice question")
    options: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of options for MCQs")
    selected_option: Optional[str] = Field(default=None, description="Selected option for MCQs")

class ExamExtraction(BaseModel):
    """Represents the full extracted exam data."""
    extracted_questions: List[ExtractedQuestion] = Field(description="List of extracted questions and answers")

class PDFExtractor:
    def __init__(self, api_key=None):
        """Initialize the PDF Extractor with a Groq API key."""
        try:
            from django.conf import settings
            self.api_key = api_key or getattr(settings, 'GROQ_API_KEY', None)
            
            if not self.api_key:
                raise ValueError("No Groq API key provided. Set GROQ_API_KEY in settings.py or pass as parameter.")
            
            # Set environment variable for LangChain
            os.environ["GROQ_API_KEY"] = self.api_key
            
        except Exception as e:
            logger.error(f"PDFExtractor initialization failed: {e}")
            raise ValueError(f"Failed to initialize PDFExtractor: {str(e)}")
    
    def extract_images_from_pdf(self, pdf_file) -> List[Tuple[int, bytes]]:
        """Extract images from each page of a PDF file object."""
        images = []
        try:
            # Create a temporary file-like object
            pdf_bytes = BytesIO(pdf_file.read())
            pdf_file.seek(0)  # Reset file pointer for potential reuse
            
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                # Log total number of pages
                logger.info(f"Total pages in PDF: {len(doc)}")
                
                for page_num, page in enumerate(doc, start=1):
                    try:
                        # Get a high-resolution pixmap (image representation) of the page
                        pix = page.get_pixmap()
                        # Convert the pixmap to bytes in PNG format
                        img_bytes = pix.tobytes("png")
                        images.append((page_num, img_bytes))
                    except Exception as page_error:
                        logger.warning(f"Error extracting image from page {page_num}: {page_error}")
        except Exception as e:
            logger.error(f"Comprehensive error extracting images from PDF: {e}")
            raise Exception(f"Error extracting images from PDF: {str(e)}")
        
        # Log number of images extracted
        logger.info(f"Successfully extracted {len(images)} page images")
        return images
    
    def encode_image_to_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes to a Base64 string."""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Process a PDF file and extract text using LangChain."""
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage
            
            images = self.extract_images_from_pdf(pdf_file)
            
            # Initialize the LLM
            llm = ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.0,
                max_tokens=4096
            )
            
            combined_text = ""
            for page_num, img_bytes in images:
                try:
                    encoded_image = self.encode_image_to_base64(img_bytes)
                    
                    # Prepare the multimodal content
                    multimodal_content = [
                        {
                            "type": "text",
                            "text": f"Extract all text from this page {page_num}. Provide the raw text content only. Be precise and comprehensive. Preserve any numbering, lettering, and formatting of multiple-choice options."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            }
                        }
                    ]
                    
                    # Create the message and invoke LLM
                    message = HumanMessage(content=multimodal_content)
                    response = llm.invoke([message])
                    
                    extracted_text = response.content
                    combined_text += f"\nPage {page_num} output:\n{extracted_text}"
                    
                    # Log successful text extraction for each page
                    logger.info(f"Successfully extracted text from page {page_num}")
                
                except Exception as page_error:
                    logger.error(f"Error processing page {page_num}: {page_error}")
                    combined_text += f"\nError processing page {page_num}: {str(page_error)}"
            
            # Log total extracted text
            logger.info(f"Total extracted text length: {len(combined_text)} characters")
            return combined_text
        
        except Exception as e:
            logger.error(f"Comprehensive text extraction error: {e}")
            raise
    
    def extract_questions_and_answers(self, pdf_file) -> List[Dict[str, Any]]:
        """
        Extract questions and answers from the PDF using LangChain with Pydantic,
        with enhanced MCQ detection.
        """
        try:
            from langchain_groq import ChatGroq
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import PydanticOutputParser
            
            # First extract all text from the PDF
            extracted_text = self.extract_text_from_pdf(pdf_file)
            
            # Validate extracted text
            if not extracted_text or len(extracted_text.strip()) < 50:
                logger.error("Insufficient text extracted from PDF")
                return []
            
            # Truncate long text to prevent API token overflow
            max_text_length = 10000
            if len(extracted_text) > max_text_length:
                logger.warning(f"Truncating extracted text from {len(extracted_text)} to {max_text_length} characters")
                extracted_text = extracted_text[:max_text_length]
            
            # Pre-analyze the text to see if it might contain MCQs
            might_contain_mcqs = self._detect_mcq_patterns(extracted_text)
            logger.info(f"MCQ pre-detection result: {might_contain_mcqs}")
            
            # Initialize the parser
            parser = PydanticOutputParser(pydantic_object=ExamExtraction)
            
            # Create a detailed prompt template with enhanced MCQ instructions
            prompt_template = PromptTemplate.from_template(
                """You are an extremely strict text extraction bot. Your ONLY goal is to extract specific pieces of text from an exam document.
                Analyze the following exam content meticulously. Identify each distinct question or problem statement.
                For each question, extract its exact identifier, the exact text of the question, and the exact block of text that constitutes the student's answer.
                
                You ABSOLUTELY MUST NOT add any extra text, commentary, summarization, or interpretation to the extracted content.
                You MUST only include text that was present in the original document within the designated question or answer fields.
                
                Format the extracted information as a single JSON object. The JSON structure MUST strictly
                adhere to the following format instructions derived from the Pydantic schema:
                {format_instructions}
                
                Guidelines for Extraction:
                - Identify each distinct question or problem presented for the student. Look for explicit numbers (e.g., 1., 2., 3.), keywords like "Question" or "Problem", or clear problem statements followed by a solution.
                - For each identified question/problem, extract its unique identifier (number, title, or part) into 'question_no'.
                - Extract the *complete, verbatim text* of the question or problem statement into 'question_statement'. 
                - Extract the *complete, verbatim block of text* from the document that represents the student's answer for this question/problem into 'complete_answer'.
                - If no clear answer is found, use "No answer provided" for the complete_answer field.
                
                VERY IMPORTANT - Multiple Choice Question (MCQ) Detection:
                - Look carefully for multiple-choice questions that have options labeled with letters (A, B, C, D) or numbers.
                - MCQ indicators include: lettered/numbered options, option lists, bubbles/checkboxes, circled answers, or marked selections.
                - Common MCQ formats include options arranged vertically with A), B), C) prefixes or horizontally like "(A) option1 (B) option2".
                - For any detected MCQ question, set the field "is_mcq" to true.
                - For MCQs, identify all available options and create an "options" array with each option having:
                  * "letter": The option identifier (A, B, C, D, etc.)
                  * "text": The full text of the option
                  * "is_selected": true if this option appears to be selected by the student, false otherwise
                - Also set "selected_option" to the letter of the selected option (if any is selected)
                - Example: A question with "A) Paris B) London C) Berlin D) Madrid" where B is circled would have is_mcq=true, all options listed, and selected_option="B"
                
                Exam Content:
                ---
                {text}
                ---
                
                JSON Output:
                """
            )
            
            # Bind the parser's format instructions to the prompt template
            prompt = prompt_template.partial(
                format_instructions=parser.get_format_instructions()
            )
            
            # Initialize the LLM
            llm = ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.0,
                max_tokens=4096
            )
            
            # Create the chain
            chain = prompt | llm | parser
            
            # Invoke the chain
            logger.info("Invoking LLM chain to extract data...")
            try:
                result: ExamExtraction = chain.invoke({"text": extracted_text})
                logger.info(f"Extracted {len(result.extracted_questions)} questions from PDF")
                
                # Log MCQ detection results
                mcq_count = sum(1 for q in result.extracted_questions if q.is_mcq)
                logger.info(f"Detected {mcq_count} MCQs out of {len(result.extracted_questions)} questions")
                
                # Convert to list of dictionaries
                questions_list = [q.model_dump() for q in result.extracted_questions]
                
                # Apply post-processing to catch any missed MCQs
                if mcq_count == 0 and might_contain_mcqs:
                    logger.info("No MCQs were detected by LLM but pre-detection found potential MCQs. Applying post-processing.")
                    questions_list = self._post_process_for_mcqs(questions_list, extracted_text)
                
                return questions_list
                
            except Exception as chain_error:
                logger.error(f"Error in LLM chain: {chain_error}")
                
                # Fallback to direct API call if LangChain parsing fails
                from groq import Groq
                
                # Construct a simpler prompt for fallback with MCQ detection
                fallback_prompt = f"""
                Extract all questions and answers from this exam paper. Pay special attention to multiple-choice questions (MCQs).
                
                For MCQs, indicate:
                - Set "is_mcq" to true
                - Include "options" as an array of choices
                - Set "selected_option" to the chosen option
                
                For all questions, extract:
                - question_no: the question number/identifier
                - question_statement: the full question text
                - complete_answer: the student's answer (or "No answer provided")
                
                Format as JSON array with these fields.
                
                Exam content:
                {extracted_text}
                """
                
                client = Groq(api_key=self.api_key)
                completion = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": fallback_prompt}],
                    temperature=0.0,
                    max_tokens=4096,
                    stream=False,
                )
                
                response_text = completion.choices[0].message.content
                
                # Try to extract JSON
                try:
                    # Try to match JSON array
                    import re
                    json_match = re.search(r'\[\s*\{.+\}\s*\]', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        result = json.loads(json_str)
                        logger.info(f"Extracted {len(result)} questions using fallback method")
                        
                        # Apply post-processing for MCQs if none were detected
                        mcq_count = sum(1 for q in result if q.get('is_mcq', False))
                        if mcq_count == 0 and might_contain_mcqs:
                            logger.info("No MCQs detected by fallback. Applying post-processing.")
                            result = self._post_process_for_mcqs(result, extracted_text)
                        
                        return result
                    else:
                        # Try parsing the whole response as JSON
                        result = json.loads(response_text)
                        if isinstance(result, list):
                            logger.info(f"Extracted {len(result)} questions from direct JSON response")
                            
                            # Apply post-processing
                            mcq_count = sum(1 for q in result if q.get('is_mcq', False))
                            if mcq_count == 0 and might_contain_mcqs:
                                result = self._post_process_for_mcqs(result, extracted_text)
                            
                            return result
                        
                        # If it's a JSON object with a questions field
                        if isinstance(result, dict) and "questions" in result:
                            logger.info(f"Extracted {len(result['questions'])} questions from JSON object")
                            return result["questions"]
                        
                        logger.error("Failed to find questions array in response")
                        return []
                        
                except Exception as json_error:
                    logger.error(f"JSON extraction error: {json_error}")
                    logger.error(f"Raw response: {response_text[:500]}...")
                    return []
                
        except Exception as e:
            logger.error(f"Comprehensive PDF processing error: {e}")
            return []
    
    def _detect_mcq_patterns(self, text: str) -> bool:
        """
        Pre-analyze text to detect if it likely contains MCQs.
        """
        # Patterns that strongly indicate MCQs
        mcq_patterns = [
            r'\([A-D]\)\s+\w+',  # (A) Option format
            r'[A-D]\)\s+\w+',    # A) Option format
            r'\b[A-D]\.\s+\w+',  # A. Option format
            r'Option\s+[A-D]',   # Option A format
            r'(?:^|\n)[\t ]*[A-D][\.\):][\t ]+\w+', # MCQ at line start
            r'(?:circle|mark|choose|select)\s+(?:one|the correct|the right|the best|your|an)?\s*(?:option|answer|choice)',  # Instruction patterns
            r'select one of the following',
            r'bubble\s+(?:in|filled|marked|selected)',  # Bubble references
            r'multiple(?:\s+|-)?choice',  # Explicit MCQ reference
        ]
        
        # Check for matches
        for pattern in mcq_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Count sequential option-like patterns
        option_sequences = re.findall(r'(?:[A-D][\.\):][\t ]+\w+[^\n]*\n){2,}', text)
        if option_sequences:
            return True
        
        return False
    
    def _post_process_for_mcqs(self, questions: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        """
        Apply post-processing to detect MCQs that might have been missed.
        """
        processed_questions = []
        
        for question in questions:
            # Skip if already identified as MCQ
            if question.get('is_mcq', False):
                processed_questions.append(question)
                continue
            
            question_statement = question.get('question_statement', '')
            complete_answer = question.get('complete_answer', '')
            
            # Check if question contains typical MCQ patterns
            mcq_patterns = [
                r'\([A-D]\)\s+\w+',  # (A) Option format
                r'[A-D]\)\s+\w+',    # A) Option format
                r'\b[A-D]\.\s+\w+',  # A. Option format
                r'(?:^|\n)[\t ]*[A-D][\.\):][\t ]+\w+', # MCQ at line start
            ]
            
            is_mcq = False
            for pattern in mcq_patterns:
                # Check both question statement and answer
                if (re.search(pattern, question_statement, re.IGNORECASE) or 
                    re.search(pattern, complete_answer, re.IGNORECASE)):
                    is_mcq = True
                    break
            
            # Special case: Look for selected option patterns in answer
            selected_option = None
            if complete_answer:
                option_match = re.search(r'(?:^|\s+)([A-D])(?:\s*$|\.|\)|\s+is\s+(?:the\s+)?(?:answer|selected|chosen))', 
                                        complete_answer, re.IGNORECASE)
                if option_match:
                    is_mcq = True
                    selected_option = option_match.group(1).upper()
            
            # Selected based on answer formatting
            if complete_answer in ["A", "B", "C", "D"]:
                is_mcq = True
                selected_option = complete_answer
            
            # If identified as MCQ, extract options
            if is_mcq:
                options = []
                
                # First try to extract from question statement
                option_text = question_statement
                if not re.search(r'[A-D][\.\):]', option_text):
                    # If not in question, try the full text around this question
                    question_no = question.get('question_no', '')
                    if question_no:
                        # Find context around question number
                        context_pattern = f"(?:{question_no}|Question\\s+{question_no}).*?(?:(?:\\n\\n)|$)"
                        context_match = re.search(context_pattern, full_text, re.DOTALL)
                        if context_match:
                            option_text = context_match.group(0)
                
                # Extract options using regex
                option_matches = re.finditer(r'([A-D])[\.\):][\t ]+([^\n]+)', option_text, re.IGNORECASE)
                for match in option_matches:
                    letter = match.group(1).upper()
                    text = match.group(2).strip()
                    is_selected = selected_option == letter if selected_option else False
                    
                    options.append({
                        "letter": letter,
                        "text": text,
                        "is_selected": is_selected
                    })
                
                # Update question with MCQ data
                question.update({
                    "is_mcq": True,
                    "options": options,
                    "selected_option": selected_option
                })
            
            processed_questions.append(question)
        
        return processed_questions