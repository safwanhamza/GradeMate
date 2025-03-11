import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust to your path

# Configuration
OCR_OUTPUT_DIR = "ocr_outputs"
VECTOR_DB_PATH = "vectorstore.db"
EMBEDDINGS_MODEL = "text-embedding-ada-002"  # OpenAI model for embeddings


def extract_text_from_image(image_path):
    """
    Extract text from an image using OCR.
    """
    try:
        # Read the image
        img = cv2.imread(image_path)

        # Preprocess image for better OCR results (optional)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # OCR to extract text
        text = pytesseract.image_to_string(binary)
        return text
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return ""


def save_extracted_text(text, filename):
    """
    Save extracted text to a file.
    """
    if not os.path.exists(OCR_OUTPUT_DIR):
        os.makedirs(OCR_OUTPUT_DIR)

    with open(os.path.join(OCR_OUTPUT_DIR, filename), "w", encoding="utf-8") as file:
        file.write(text)


def preprocess_text(text):
    """
    Perform basic text preprocessing: remove extra spaces, normalize text, etc.
    """
    # Strip unwanted whitespace
    cleaned_text = text.strip()

    # Optionally, additional normalization can be added here
    cleaned_text = " ".join(cleaned_text.split())  # Normalize whitespace

    return cleaned_text


def split_text_to_documents(text, chunk_size=512, chunk_overlap=100):
    """
    Split text into smaller, manageable chunks for embeddings.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents([Document(page_content=text)])


def store_in_vector_db(documents):
    """
    Store documents into a vector database using embeddings.
    """
    # Initialize embeddings model
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)

    # Check if vector DB exists
    if os.path.exists(VECTOR_DB_PATH):
        vector_db = FAISS.load_local(VECTOR_DB_PATH, embeddings)
    else:
        vector_db = FAISS(embeddings=embeddings)

    # Add documents to vector DB
    vector_db.add_documents(documents)

    # Save vector DB
    vector_db.save_local(VECTOR_DB_PATH)
    print(f"Documents successfully stored in vector DB at {VECTOR_DB_PATH}.")


def process_exam_script(image_path):
    """
    End-to-end processing of an exam script image.
    """
    print(f"Processing: {image_path}")

    # Step 1: OCR - Extract text from image
    raw_text = extract_text_from_image(image_path)
    if not raw_text:
        return

    # Step 2: Save raw text (optional)
    text_filename = os.path.basename(image_path).replace(".jpg", ".txt").replace(".png", ".txt")
    save_extracted_text(raw_text, text_filename)

    # Step 3: Preprocess text
    cleaned_text = preprocess_text(raw_text)

    # Step 4: Split into smaller documents
    documents = split_text_to_documents(cleaned_text)

    # Step 5: Store in vector DB
    store_in_vector_db(documents)


# Driver code
def main():
    input_dir = "exam_scripts"  # Directory containing scanned exam script images

    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist. Please create it and add image files.")
        return

    for image_file in os.listdir(input_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_file)
            process_exam_script(image_path)


if __name__ == "__main__":
    main()
