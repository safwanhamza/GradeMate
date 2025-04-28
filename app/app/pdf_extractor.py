# pdf_extractor.py
import os
import base64
import fitz  # PyMuPDF
from io import BytesIO
from typing import List, Tuple, Dict, Any
import json
import re
import logging
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from datetime import datetime

# Configure logging with more robust settings
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to reduce noise
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f"pdf_extractor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self, api_key=None):
        """Initialize the PDF Extractor with a Groq API key."""
        try:
            from django.conf import settings
            self.api_key = api_key or getattr(settings, 'GROQ_API_KEY', None)
            
            if not self.api_key:
                raise ValueError("No Groq API key provided. Set GROQ_API_KEY in settings.py or pass as parameter.")
            
            # Initialize the Groq client directly
            self.client = Groq(api_key=self.api_key)
            
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
        """Process a PDF file and extract text using Groq LLM."""
        try:
            images = self.extract_images_from_pdf(pdf_file)
            
            combined_text = ""
            for page_num, img_bytes in images:
                try:
                    encoded_image = self.encode_image_to_base64(img_bytes)
                    
                    # Prepare the multimodal content for the user message
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
                    
                    # Use the direct Groq client to call the API
                    completion = self.client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=[{"role": "user", "content": multimodal_content}],
                        temperature=0.0,
                        max_tokens=4096,
                        stream=False,
                    )
                    
                    extracted_text = completion.choices[0].message.content
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
        Extract questions and answers from the PDF with comprehensive error handling.
        """
        try:
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
            
            # Construct a more robust prompt
            prompt = f"""
            Carefully extract all questions and answers from this exam paper. 
            Follow these strict guidelines:
            1. Identify EACH question's number precisely
            2. Extract the COMPLETE question text verbatim
            3. Extract the CORRESPONDING student's answer
            4. If no clear answer is found, use "No answer provided"
            
            IMPORTANT: Return a valid JSON array. If NO questions are found, return an empty array.
            
            JSON Structure:
            [
                {{
                    "question_no": "Question number as string",
                    "question_statement": "Exact question text",
                    "complete_answer": "Exact student answer or 'No answer provided'"
                }}
            ]
            
            Exam content:
            {extracted_text}
            """
            
            try:
                # Use the Groq client with enhanced parameters
                completion = self.client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # Most deterministic setting
                    max_tokens=4096,
                    stream=False,
                )
                
                # Extract the JSON response
                json_str = completion.choices[0].message.content
                
                # Robust JSON extraction function
                def extract_json(text):
                    # List of methods to extract JSON
                    extraction_methods = [
                        # Method 1: Regex for JSON array
                        lambda t: json.loads(re.search(r'(\[.*?\])', t, re.DOTALL | re.MULTILINE).group(1)) if re.search(r'(\[.*?\])', t, re.DOTALL | re.MULTILINE) else None,
                        
                        # Method 2: Direct JSON parsing
                        lambda t: json.loads(t),
                        
                        # Method 3: Strip code block markers
                        lambda t: json.loads(t.replace('```json', '').replace('```', '').strip())
                    ]
                    
                    for method in extraction_methods:
                        try:
                            result = method(text)
                            if result is not None:
                                return result
                        except (json.JSONDecodeError, AttributeError):
                            continue
                    
                    return []
                
                # Extract and validate the result
                result = extract_json(json_str)
                
                # Log extraction details
                logger.info(f"Extracted {len(result)} questions from PDF")
                
                return result
            
            except Exception as json_error:
                logger.error(f"JSON extraction error: {json_error}")
                logger.error(f"Problematic JSON string: {json_str}")
                return []
        
        except Exception as e:
            logger.error(f"Comprehensive PDF processing error: {e}")
            return []