# chroma_direct.py
import os
import shutil
import logging
import time
import fitz  # PyMuPDF
from io import BytesIO
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDirect:
    def __init__(self, embedding_model=None):
        """Initialize ChromaDirect with an optional embedding model."""
        self.embedding_model = embedding_model
        if not self.embedding_model:
            # Import a default embedding model
            try:
                from langchain_nomic.embeddings import NomicEmbeddings
                self.embedding_model = NomicEmbeddings(model="nomic-embed-text-v1")
            except ImportError:
                # Fallback to a different embedding model
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
        
        # Default DB path is in the current working directory
        self.db_path = os.path.join(os.getcwd(), "chroma_db")
        
    def get_client(self):
        """Get a client for the Chroma DB."""
        if not os.path.exists(self.db_path):
            # Create a new empty DB
            os.makedirs(self.db_path, exist_ok=True)
            client = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_model
            )
        else:
            # Load existing DB
            client = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_model
            )
        return client
    
    def reset_db(self):
        """Reset the Chroma DB."""
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        os.makedirs(self.db_path, exist_ok=True)
        return {"status": "success", "message": f"Chroma DB reset at {self.db_path}"}
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF."""
        text = ""
        try:
            # Create a temporary file-like object
            pdf_bytes = BytesIO(pdf_file.read())
            pdf_file.seek(0)  # Reset file pointer for potential reuse
            
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
                    text += "\n\n"
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
        return text
    
    def chunk_text(self, text, chunk_size=1000, chunk_overlap=200):
        """Chunk text into smaller pieces."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        return chunks
    
    def add_pdf_to_chroma(self, pdf_file, metadata=None):
        """Extract text from PDF, chunk it, and add to Chroma DB."""
        if metadata is None:
            metadata = {}
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_file)
        
        # Chunk text
        chunks = self.chunk_text(text)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata["chunk_id"] = i
            doc_metadata["source"] = pdf_file.name if hasattr(pdf_file, "name") else "unknown"
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        # Get Chroma client
        client = self.get_client()
        
        # Add documents to Chroma
        client.add_documents(documents)
        client.persist()
        
        return {
            "status": "success",
            "chunks": len(chunks),
            "source": metadata.get("source", "unknown")
        }
    
    def process_drive_files(self, files, progress_callback=None):
        """Process a list of files from Google Drive."""
        results = []
        
        for file in files:
            if progress_callback:
                progress_callback(f"Processing {file.get('name', 'unknown file')}")
            
            try:
                # Download file
                file_id = file.get("id")
                file_name = file.get("name", "unknown")
                
                # Import here to avoid circular imports
                from .drive_utils import download_drive_file
                
                file_stream = download_drive_file(file_id)
                
                # Add file to Chroma
                metadata = {
                    "source": file_name,
                    "drive_id": file_id,
                    "type": "pdf"
                }
                
                result = self.add_pdf_to_chroma(file_stream, metadata)
                results.append(result)
                
                if progress_callback:
                    progress_callback(f"✅ Added {file_name} to Chroma DB with {result.get('chunks', 0)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file.get('name', 'unknown')}: {e}")
                if progress_callback:
                    progress_callback(f"❌ Error processing {file.get('name', 'unknown')}: {e}")
                
                results.append({
                    "status": "error",
                    "source": file.get("name", "unknown"),
                    "error": str(e)
                })
        
        return results
    
    def search_documents(self, query, k=5):
        """Search for documents relevant to the query."""
        client = self.get_client()
        
        retriever = client.as_retriever(
            search_type="mmr",
            search_kwargs={"fetch_k": k*2, "k": k}
        )
        
        docs = retriever.get_relevant_documents(query)
        return docs