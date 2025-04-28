#drive_fetch_test.py

import os
import fitz  # PyMuPDF
import sqlite3
import time
import tempfile
import base64
import shutil
import zipfile
import logging
from io import BytesIO
from typing import List, Dict, Any
from datetime import datetime
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import gdown
import pandas as pd
import torch
from groq import Groq
from PyPDF2 import PdfReader

from langchain_nomic.embeddings import NomicEmbeddings
from langchain.schema import Document
from langchain.text_splitter import ( CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter, HTMLHeaderTextSplitter )
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_community.vectorstores import Chroma

from unstructured.partition.auto import partition
from unstructured.documents.elements import (Image as UnstructuredImage, Table as UnstructuredTable, Text, Title, ListItem, NarrativeText)

# Settings
NOMIC_API_KEY = "nk-rjqDQKJtuoRaTcocIvaSJr6g5JItcyLvNJR4O7h153o"
GROQ_API_KEY = "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"
DRIVE_FOLDER_ID = "1wEtrJspJlZuYiNMVf584sLlVs72c7CgT"

DOWNLOAD_DIR = "downloaded_files"
FIGURES_DIR = "extracted_figures"
DB_PATH = "textbooks.db"
PERSIST_DIR = "chroma_db"

# Path where your client secret JSON is
CLIENT_SECRET_PATH = "client_secret_203302872030-p9gkft6otmaaubsroa4ob1k3310lfen9.apps.googleusercontent.com.json"


os.environ["ATLAS_API_KEY"] = NOMIC_API_KEY

client = Groq(api_key=GROQ_API_KEY)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f"drive_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TEXT_TYPES = (Text, Title, ListItem, NarrativeText)

def authenticate_gdrive():
    gauth = GoogleAuth()
    
    gauth.LoadClientConfigFile(CLIENT_SECRET_PATH)

    # Try to load existing credentials
    if os.path.exists("mycreds.txt"):
        gauth.LoadCredentialsFile("mycreds.txt")

    if gauth.credentials is None:
        # First time login
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # If expired
        gauth.Refresh()
    else:
        # Already valid
        gauth.Authorize()
    
    gauth.SaveCredentialsFile("mycreds.txt")
    drive = GoogleDrive(gauth)
    return drive

def fetch_from_drive() -> List[Dict[str, Any]]:
    results = []
    logger.info(f"Fetching files from Google Drive folder: {DRIVE_FOLDER_ID}")

    drive = authenticate_gdrive()

    query = f"'{DRIVE_FOLDER_ID}' in parents and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()

    for file in file_list:
        file_id = file['id']
        file_name = file['title']

        # ✅ Only allow PDF for now (later you can tell me to enable docx, txt etc)
        if not file_name.lower().endswith(".pdf"):
            logger.warning(f"Skipping unsupported file: {file_name}")
            continue

        local_path = os.path.join(DOWNLOAD_DIR, file_name)
        file.GetContentFile(local_path)

        results.append({
            "filename": file_name,
            "local_path": local_path,
            "drive_id": file_id,
        })

        logger.info(f"✅ Downloaded and saved: {file_name}")

    logger.info(f"Fetch results: {len(results)} files processed successfully.")
    return results


def setup_directories():
    for directory in [DOWNLOAD_DIR, FIGURES_DIR, PERSIST_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS drive_pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            drive_file_id TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            order_index INTEGER NOT NULL,
            embedding_id TEXT,
            FOREIGN KEY (pdf_id) REFERENCES drive_pdfs(id)
        )''')
    conn.commit()
    conn.close()
    logger.info("Database setup complete.")

def fetch_from_drive() -> List[Dict[str, Any]]:
    results = []
    logger.info(f"Fetching files from Google Drive folder: {DRIVE_FOLDER_ID}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_list = gdown.download_folder(
                f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}",
                output=tmpdir,
                quiet=False,
                use_cookies=False
            )

            if not file_list:
                logger.warning("No files found or download failed.")
                return results

            for local_path in file_list:
                filename = os.path.basename(local_path)
                dest_path = os.path.join(DOWNLOAD_DIR, filename)

                try:
                    if not os.path.isfile(local_path):
                        continue

                    shutil.copy2(local_path, dest_path)
                    logger.info(f"Downloaded and copied: {filename}")

                    results.append({
                        "filename": filename,
                        "local_path": dest_path,
                        "drive_id": filename,
                    })

                except Exception as e:
                    logger.error(f"Failed to process file {filename}: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error fetching from drive: {e}")

    logger.info(f"Fetch results: {len(results)} files processed successfully.")
    return results

def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.error(f"PDF extraction failed for {path}: {e}")
        return ""

def partition_file(path, output_figures_dir):
    file_extension = os.path.splitext(path)[1].lower()
    
    # Process different file types
    try:
        # Try unstructured's partition for supported files first
        try:
            return partition(
                filename=path,
                strategy="hi_res",
                languages=["eng"],
                extract_image_block_to_payload=True,
                extract_image_block_types=["Image", "Table"],
                infer_table_structure=True,
                image_output_dir_path=output_figures_dir
            )
        except ImportError:
            logger.warning(f"Unstructured partition not available. Using alternative methods for {path}")
            
            # Handle PDF files with PyMuPDF
            if file_extension == '.pdf':
                return handle_pdf_with_pymupdf(path, output_figures_dir)
            
            # Handle CSV files
            elif file_extension == '.csv':
                return handle_csv(path)
            
            # Handle HTML files
            elif file_extension == '.html':
                return handle_html(path)
            
            # Handle Jupyter notebooks
            elif file_extension == '.ipynb':
                return handle_ipynb(path)
            
            # Handle Python files
            elif file_extension == '.py':
                return handle_python(path)
            
            # Handle plain text and other text-based files
            elif file_extension in ['.txt', '.md', '.json', '.xml']:
                return handle_text_file(path)
            
            # Fallback to basic text extraction
            else:
                extracted_text = extract_text_from_pdf(path)
                return [NarrativeText(text=extracted_text)]
    except Exception as e:
        logger.error(f"Error processing file {path}: {e}")
        return [NarrativeText(text=f"Error processing file: {str(e)}")] 












def check_vectordb_content():
    # Load the existing Chroma DB

    
    embedding_model = NomicEmbeddings(model="nomic-embed-text-v1")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    
    # Get all documents
    docs = vectordb.get()
    
    print(f"Total documents in vector database: {len(docs['ids'])}")
    
    # Sample 5 documents
    for i in range(min(5, len(docs['ids']))):
        doc_id = docs['ids'][i]
        doc_content = docs['documents'][i]
        print(f"\nDocument ID: {doc_id}")
        print(f"Content (truncated): {doc_content[:200]}...")
        
    # Get all metadatas
    metadatas = docs['metadatas']
    print("\nMetadata types available:")
    if len(metadatas) > 0:
        metadata_keys = set()
        for metadata in metadatas:
            if metadata is not None:
                metadata_keys.update(metadata.keys())
        print(metadata_keys)

# Call this function at the end of your main function or as a separate command
# check_vectordb_content()

def check_generated_captions(image_captions, table_captions):
    print("\n=== Generated Image Captions ===")
    for img_id, caption in image_captions.items():
        print(f"Image ID: {img_id[:10]}...")
        print(f"Caption: {caption}\n")
    
    print("\n=== Generated Table Captions ===")
    for table_id, caption in table_captions.items():
        print(f"Table ID: {table_id[:10]}...")
        print(f"Caption: {caption}\n")
# Modify your main function to include this after caption generation
# image_captions, table_captions = generate_captions(elements)
# check_generated_captions(image_captions, table_captions)



def create_image_caption_report(file_info, elements, image_captions):
    import os
    from datetime import datetime
    
    report_dir = "image_caption_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    filename = file_info["filename"]
    base_name = os.path.splitext(filename)[0]
    
    report_path = os.path.join(report_dir, f"{base_name}_report.html")
    
    with open(report_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Image Caption Report: {filename}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .image-container {{ border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; }}
                img {{ max-width: 100%; max-height: 300px; }}
                .caption {{ background-color: #f8f9fa; padding: 10px; margin-top: 10px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1>Image Caption Report: {filename}</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """)
        
        # Find all images with captions
        image_count = 0
        for el in elements:
            if hasattr(el, 'id') and el.id in image_captions:
                image_count += 1
                
                # Get image path from metadata if available
                img_path = ""
                if hasattr(el, 'metadata') and isinstance(el.metadata, dict) and 'filename' in el.metadata:
                    img_path = el.metadata['filename']
                
                # Get caption
                caption = image_captions[el.id]
                
                # Write to report
                f.write(f"""
                <div class="image-container">
                    <h3>Image {image_count}</h3>
                """)
                
                if os.path.exists(img_path):
                    f.write(f'<img src="file:///{img_path.replace("\\", "/")}" alt="Image {image_count}">')
                else:
                    f.write(f'<p>Image file not found: {img_path}</p>')
                
                f.write(f"""
                    <div class="caption">
                        <strong>Caption:</strong> {caption}
                    </div>
                </div>
                """)
        
        f.write("""
        </body>
        </html>
        """)
    
    print(f"Generated image caption report at: {report_path}")
    return report_path

# Add to your main function after generating captions
# report_path = create_image_caption_report(file_info, elements, image_captions)

def check_chunks_correctness(filename, chunks):
    print(f"\n=== Checking Chunks for {filename} ===")
    print(f"Total chunks: {len(chunks)}")
    
    # Print a few sample chunks
    for i in range(min(3, len(chunks))):
        print(f"\nChunk {i+1}:")
        print(f"{chunks[i][:200]}..." if len(chunks[i]) > 200 else chunks[i])
    
    # Check for potential issues
    empty_chunks = [i for i, chunk in enumerate(chunks) if not chunk.strip()]
    very_short_chunks = [i for i, chunk in enumerate(chunks) if len(chunk.strip()) < 20]
    
    if empty_chunks:
        print(f"\nWarning: Found {len(empty_chunks)} empty chunks at indices: {empty_chunks}")
    
    if very_short_chunks:
        print(f"\nWarning: Found {len(very_short_chunks)} very short chunks at indices: {very_short_chunks}")

# Modify your code to check chunks before storing in DB
# texts = [doc.page_content for doc in chunked_docs]
# check_chunks_correctness(filename, texts)
# store_in_db(filename, file_info["drive_id"], texts)










# PyMuPDF for PDF handling
def handle_pdf_with_pymupdf(path, output_figures_dir):
    logger.warning(f"Using PyMuPDF fallback method for PDF extraction: {path}")
    try:
        elements = []
        doc = fitz.open(path)
        
        # Process each page
        for page_num, page in enumerate(doc):
            # Extract text
            text = page.get_text()
            if text.strip():
                elements.append(NarrativeText(text=text))
            
            # Extract images if needed
            image_count = 0
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save the image
                    img_filename = f"page{page_num+1}_img{img_index+1}.png"
                    img_path = os.path.join(output_figures_dir, img_filename)
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Create image element - fixed approach
                    image_element = UnstructuredImage(
                        text=f"[Image from page {page_num+1}, index {img_index}]"
                    )
                    # Add metadata separately
                    image_element.metadata = {"filename": img_path}
                    
                    elements.append(image_element)
                    image_count += 1
                except Exception as e:
                    logger.error(f"Error extracting image: {e}")
            
            logger.info(f"Extracted {image_count} images from page {page_num+1}")
        
        doc.close()
        return elements
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        
        # Ultimate fallback - just get basic text with PyPDF2
        extracted_text = extract_text_from_pdf(path)
        return [NarrativeText(text=extracted_text)]


# CSV handling
def handle_csv(path):
    logger.info(f"Processing CSV file: {path}")
    try:
        import pandas as pd
        
        df = pd.read_csv(path)
        elements = []
        
        # Add metadata about the CSV
        csv_meta = f"CSV file with {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns.tolist())}"
        elements.append(NarrativeText(text=csv_meta))
        
        # Add sample data (first few rows)
        sample_data = f"Sample data: {df.head(5).to_string()}"
        elements.append(NarrativeText(text=sample_data))
        
        return elements
    except Exception as e:
        logger.error(f"Error processing CSV file {path}: {e}")
        return [NarrativeText(text=f"Error processing CSV file: {str(e)}")]

# HTML handling
def handle_html(path):
    logger.info(f"Processing HTML file: {path}")
    try:
        from bs4 import BeautifulSoup
        
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title_text = soup.title.string if soup.title else "No title"
        
        # Extract all text content
        body_text = soup.get_text(separator='\n', strip=True)
        
        elements = [
            Title(text=f"HTML Document: {title_text}"),
            NarrativeText(text=body_text)
        ]
        
        return elements
    except Exception as e:
        logger.error(f"Error processing HTML file {path}: {e}")
        return [NarrativeText(text=f"Error processing HTML file: {str(e)}")]

# Jupyter notebook handling
def handle_ipynb(path):
    logger.info(f"Processing Jupyter notebook: {path}")
    try:
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        elements = []
        
        # Add notebook metadata
        if 'metadata' in notebook and notebook['metadata']:
            meta_text = f"Jupyter notebook metadata: {json.dumps(notebook['metadata'])}"
            elements.append(NarrativeText(text=meta_text))
        
        # Process cells
        for i, cell in enumerate(notebook.get('cells', [])):
            cell_type = cell.get('cell_type', '')
            
            if cell_type == 'markdown':
                source = ''.join(cell.get('source', []))
                elements.append(NarrativeText(text=f"Markdown Cell {i+1}: {source}"))
            
            elif cell_type == 'code':
                source = ''.join(cell.get('source', []))
                elements.append(NarrativeText(text=f"Code Cell {i+1}: {source}"))
                
                # Include outputs if available
                outputs = cell.get('outputs', [])
                for output in outputs:
                    if 'text' in output:
                        output_text = ''.join(output['text'])
                        elements.append(NarrativeText(text=f"Output: {output_text}"))
                    elif 'data' in output and 'text/plain' in output['data']:
                        output_text = ''.join(output['data']['text/plain'])
                        elements.append(NarrativeText(text=f"Output: {output_text}"))
        
        return elements
    except Exception as e:
        logger.error(f"Error processing Jupyter notebook {path}: {e}")
        return [NarrativeText(text=f"Error processing Jupyter notebook: {str(e)}")]

# Python file handling
def handle_python(path):
    logger.info(f"Processing Python file: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple tokenization of Python code into logical sections
        elements = []
        
        # Add file info
        elements.append(Title(text=f"Python File: {os.path.basename(path)}"))
        
        # Split code by functions/classes for better chunking
        import re
        
        # Extract imports section
        imports_pattern = r'^import.*|^from.*import'
        imports = re.findall(imports_pattern, content, re.MULTILINE)
        if imports:
            imports_text = '\n'.join(imports)
            elements.append(NarrativeText(text=f"Imports:\n{imports_text}"))
        
        # Extract class definitions
        class_pattern = r'(class\s+\w+.*?:(?:\s*"""[\s\S]*?""")?(?:\s*#[^\n]*)?(?:\s*@.*)?(?:\s*def\s+\w+\([^)]*\)\s*:[\s\S]*?)?)(?=\n\s*class|\n\s*def|\Z)'
        classes = re.findall(class_pattern, content, re.DOTALL)
        for i, class_def in enumerate(classes):
            elements.append(NarrativeText(text=f"Class Definition {i+1}:\n{class_def}"))
        
        # Extract function definitions
        func_pattern = r'(def\s+\w+\([^)]*\)\s*:(?:\s*"""[\s\S]*?""")?[\s\S]*?)(?=\n\s*def|\n\s*class|\Z)'
        funcs = re.findall(func_pattern, content, re.DOTALL)
        for i, func_def in enumerate(funcs):
            elements.append(NarrativeText(text=f"Function {i+1}:\n{func_def}"))
        
        # Add other code that's not in functions/classes
        other_code = content
        for class_def in classes:
            other_code = other_code.replace(class_def, '')
        for func_def in funcs:
            other_code = other_code.replace(func_def, '')
        
        # Remove import statements from other code
        for imp in imports:
            other_code = other_code.replace(imp, '')
        
        # If there's remaining code, add it
        other_code = other_code.strip()
        if other_code:
            elements.append(NarrativeText(text=f"Other Code:\n{other_code}"))
        
        return elements
    except Exception as e:
        logger.error(f"Error processing Python file {path}: {e}")
        return [NarrativeText(text=f"Error processing Python file: {str(e)}")]

# Plain text file handling
def handle_text_file(path):
    logger.info(f"Processing text file: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        elements = [NarrativeText(text=content)]
        return elements
    except UnicodeDecodeError:
        try:
            # Try with a different encoding
            with open(path, 'r', encoding='latin-1') as f:
                content = f.read()
            elements = [NarrativeText(text=content)]
            return elements
        except Exception as e:
            logger.error(f"Error processing text file {path} with latin-1 encoding: {e}")
            return [NarrativeText(text=f"Error processing text file: {str(e)}")]
    except Exception as e:
        logger.error(f"Error processing text file {path}: {e}")
        return [NarrativeText(text=f"Error processing text file: {str(e)}")]


def generate_captions(elements):
    image_captions, table_captions = {}, {}

    for el in elements:
        if isinstance(el, (UnstructuredImage, UnstructuredTable)):
            prompt = "Generate a caption for a scientific diagram or table."

            try:
                completion = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_completion_tokens=100,
                    top_p=1,
                    stream=False,
                    stop=None,
                )
                caption = completion.choices[0].message.content.strip()

                if isinstance(el, UnstructuredImage):
                    image_captions[el.id] = caption
                else:
                    table_captions[el.id] = caption

            except Exception as e:
                logger.warning(f"Caption generation failed: {e}")

    return image_captions, table_captions

def convert_to_documents(elements, image_captions, table_captions):
    docs = []
    for el in elements:
        if isinstance(el, TEXT_TYPES):
            docs.append(Document(page_content=el.text, metadata={"type": "text", "element_id": el.id}))
        elif isinstance(el, UnstructuredImage) and el.id in image_captions:
            docs.append(Document(page_content=f"[IMAGE] {image_captions[el.id]}", metadata={"type": "image_caption", "element_id": el.id}))
        elif isinstance(el, UnstructuredTable) and el.id in table_captions:
            docs.append(Document(page_content=f"[TABLE] {table_captions[el.id]}", metadata={"type": "table_caption", "element_id": el.id}))
    return docs

def get_splitter(doc_type: str):
    if doc_type == "markdown":
        return MarkdownTextSplitter(chunk_size=512, chunk_overlap=64)
    elif doc_type == "html":
        return HTMLHeaderTextSplitter(headers_to_split_on=["h1", "h2", "h3"], chunk_size=800, chunk_overlap=100)
    elif doc_type == "json":
        return RecursiveJsonSplitter(max_chunk_size=500, min_chunk_size=100)
    elif doc_type == "code":
        return RecursiveCharacterTextSplitter.from_language("python", chunk_size=1000, chunk_overlap=200)
    else:
        return CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def chunk_documents(docs: List[Document]) -> List[Document]:
    chunks = []
    for doc in docs:
        splitter = get_splitter(doc.metadata.get("doc_type", "text"))
        chunks.extend(splitter.split_documents([doc]))
    return chunks

def store_in_db(title: str, drive_id: str, chunks: List[str]):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO drive_pdfs (title, drive_file_id) VALUES (?, ?)", (title, drive_id))
    pdf_id = cursor.lastrowid
    chunk_data = [(pdf_id, chunk, idx) for idx, chunk in enumerate(chunks)]
    cursor.executemany("INSERT INTO text_chunks (pdf_id, content, order_index) VALUES (?, ?, ?)", chunk_data)
    conn.commit()
    conn.close()
    logger.info(f"Stored {len(chunks)} chunks for {title}.")

def main():
    setup_directories()
    setup_database()

    embedding_model = NomicEmbeddings(model="nomic-embed-text-v1")
    files = fetch_from_drive()

    all_final_chunks = []

    for file_info in files:
        filename, path = file_info["filename"], file_info["local_path"]
        figures_dir = os.path.join(FIGURES_DIR, os.path.splitext(filename)[0])
        os.makedirs(figures_dir, exist_ok=True)

        # Expanded list of supported file types
        if filename.lower().endswith((".pdf", ".docx", ".txt", ".md", ".csv", ".html", ".py", ".ipynb", ".json", ".xml")):
            elements = partition_file(path, figures_dir)
            
            # Generate and check captions
            image_captions, table_captions = generate_captions(elements)
            check_generated_captions(image_captions, table_captions)
            
            # Create image caption report if there are images
            if image_captions:
                report_path = create_image_caption_report(file_info, elements, image_captions)
            
            docs = convert_to_documents(elements, image_captions, table_captions)
            chunked_docs = chunk_documents(docs)

            # Check chunks before storing
            texts = [doc.page_content for doc in chunked_docs]
            check_chunks_correctness(filename, texts)
            
            store_in_db(filename, file_info["drive_id"], texts)
            all_final_chunks.extend(chunked_docs)
        else:
            logger.warning(f"Skipping unsupported file: {filename}")

    if all_final_chunks:
        Chroma.from_documents(all_final_chunks, embedding=embedding_model, persist_directory=PERSIST_DIR)
        logger.info("Chroma vectorstore created and persisted.")
        
    # Check vector DB content at the end
    check_vectordb_content()

if __name__ == "__main__":
    main()
