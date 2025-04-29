# pdf_extractor.py with LangChain integration

import os
import base64
import fitz  # PyMuPDF
from io import BytesIO
from typing import List, Tuple, Dict, Any
import json
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
                            "text": f"Extract all text from this page {page_num}. Provide the raw text content only. Be precise and comprehensive."
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
        Extract questions and answers from the PDF using LangChain with Pydantic.
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
            
            # Initialize the parser
            parser = PydanticOutputParser(pydantic_object=ExamExtraction)
            
            # Create a detailed prompt template
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
                
                # Convert to list of dictionaries
                questions_list = [q.model_dump() for q in result.extracted_questions]
                return questions_list
                
            except Exception as chain_error:
                logger.error(f"Error in LLM chain: {chain_error}")
                
                # Fallback to direct API call if LangChain parsing fails
                from groq import Groq
                
                # Construct a simpler prompt for fallback
                fallback_prompt = f"""
                Extract all questions and answers from this exam paper.
                Format as JSON array with fields:
                - question_no: the question number/identifier
                - question_statement: the full question text
                - complete_answer: the student's answer (or "No answer provided")
                
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
                        return result
                    else:
                        # Try parsing the whole response as JSON
                        result = json.loads(response_text)
                        if isinstance(result, list):
                            logger.info(f"Extracted {len(result)} questions from direct JSON response")
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