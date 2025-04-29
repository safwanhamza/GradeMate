# file_utils.py
import os
import logging
from unstructured.documents.elements import NarrativeText

logger = logging.getLogger(__name__)

def get_supported_extension(filename):
    """
    Check if file extension is supported by our processing pipeline
    """
    extensions = {
        # Document formats
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'doc',
        '.txt': 'text',
        '.md': 'markdown',
        
        # Structured data
        '.csv': 'csv',
        '.json': 'json',
        '.xml': 'xml',
        
        # Web
        '.html': 'html',
        '.htm': 'html',
        
        # Code
        '.py': 'python',
        '.ipynb': 'jupyter'
    }
    
    ext = os.path.splitext(filename)[1].lower()
    return extensions.get(ext)

def handle_csv_file(path):
    """Handle CSV files"""
    "other will be implemented lately"
    try:
        import pandas as pd
        df = pd.read_csv(path)
        metadata = f"CSV with {len(df)} rows and {len(df.columns)} columns."
        sample = df.head(3).to_string()
        return [NarrativeText(text=f"{metadata}\n\nSample data:\n{sample}")]
    except Exception as e:
        logger.error(f"Error processing CSV {path}: {e}")
        return [NarrativeText(text=f"Error processing CSV file: {str(e)}")]

def handle_jupyter_notebook(path):
    """Handle Jupyter notebook files"""
    try:
        import json
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        elements = []
        for cell in nb.get('cells', []):
            if 'source' in cell:
                text = ''.join(cell.get('source', []))
                cell_type = cell.get('cell_type', 'unknown')
                elements.append(NarrativeText(text=f"[{cell_type.upper()}]\n{text}"))
        
        return elements if elements else [NarrativeText(text="Empty notebook or no valid content found.")]
    except Exception as e:
        logger.error(f"Error processing notebook {path}: {e}")
        return [NarrativeText(text=f"Error processing Jupyter notebook: {str(e)}")]

def handle_python_file(path):
    """Handle Python files"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [NarrativeText(text=content)]
    except Exception as e:
        logger.error(f"Error processing Python file {path}: {e}")
        return [NarrativeText(text=f"Error processing Python file: {str(e)}")]

def handle_html_file(path):
    """Handle HTML files"""
    try:
        from bs4 import BeautifulSoup
        with open(path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
        
        title = soup.title.string if soup.title else "No title"
        body = soup.get_text(separator='\n', strip=True)
        
        return [
            NarrativeText(text=f"HTML Title: {title}"),
            NarrativeText(text=body)
        ]
    except Exception as e:
        logger.error(f"Error processing HTML file {path}: {e}")
        return [NarrativeText(text=f"Error processing HTML file: {str(e)}")]

def handle_unsupported_file(path):
    """
    Fallback handler for unsupported file types
    """
    ext = os.path.splitext(path)[1].lower()
    
    try:
        # Based on extension, use appropriate handler
        if ext == '.csv':
            return handle_csv_file(path)
        elif ext == '.ipynb':
            return handle_jupyter_notebook(path)
        elif ext == '.py':
            return handle_python_file(path)
        elif ext in ['.html', '.htm']:
            return handle_html_file(path)
        elif ext in ['.txt', '.md', '.json', '.xml']:
            with open(path, 'r', encoding='utf-8') as f:
                return [NarrativeText(text=f.read())]
        else:
            # Try to extract text from PDFs
            from PyPDF2 import PdfReader
            try:
                reader = PdfReader(path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                return [NarrativeText(text=text)]
            except:
                return [NarrativeText(text=f"Unable to process file: {os.path.basename(path)}")]
    except Exception as e:
        logger.error(f"Fallback processing failed for {path}: {e}")
        return [NarrativeText(text=f"Unable to process file: {os.path.basename(path)}. Error: {str(e)}")]