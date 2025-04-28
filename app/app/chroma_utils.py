# chroma_utils.py
"""
Utility functions for managing and debugging Chroma DB in the Django application.
"""

import os
import json
import shutil
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

def verify_chroma_exists() -> Dict[str, Any]:
    """
    Check if Chroma DB exists and return information about it.
    
    Returns:
        Dict containing information about the Chroma DB
    """
    # Construct the path to the Chroma DB
    base_dir = os.getcwd()
    chroma_path = os.path.join(base_dir, "chroma_db")
    
    exists = os.path.exists(chroma_path)
    
    result = {
        "exists": exists,
        "path": chroma_path,
        "size_mb": 0,
        "file_count": 0,
        "files": []
    }
    
    if exists:
        # Calculate total size
        total_size = 0
        file_count = 0
        file_list = []
        
        for dirpath, dirnames, filenames in os.walk(chroma_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                file_size = os.path.getsize(fp)
                total_size += file_size
                file_count += 1
                
                # Add to file list (limit to first 10 to avoid too much detail)
                if len(file_list) < 10:
                    file_list.append({
                        "name": f,
                        "path": os.path.relpath(fp, chroma_path),
                        "size_kb": round(file_size / 1024, 2)
                    })
        
        result["size_mb"] = round(total_size / (1024 * 1024), 2)
        result["file_count"] = file_count
        result["files"] = file_list
    
    return result

def reset_chroma_db() -> Dict[str, Any]:
    """
    Forcibly delete and recreate the Chroma DB directory.
    
    Returns:
        Dict containing the result of the operation
    """
    chroma_path = os.path.join(os.getcwd(), "chroma_db")
    
    result = {
        "success": False,
        "message": "",
        "path": chroma_path
    }
    
    try:
        # Delete if exists
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            result["message"] = f"Successfully deleted Chroma DB at {chroma_path}"
        else:
            result["message"] = f"No Chroma DB found at {chroma_path}"
        
        # Create empty directory
        os.makedirs(chroma_path, exist_ok=True)
        result["success"] = True
        
    except Exception as e:
        result["message"] = f"Error resetting Chroma DB: {str(e)}"
        logger.exception("Error in reset_chroma_db")
    
    return result

def load_chroma_client(persist_directory: Optional[str] = None) -> Any:
    """
    Load a Chroma client and return it for direct use.
    
    Args:
        persist_directory: Optional path to the Chroma DB directory
        
    Returns:
        Chroma client instance if successful, None otherwise
    """
    try:
        from langchain_community.vectorstores import Chroma
        
        # Use default path if none provided
        if persist_directory is None:
            persist_directory = os.path.join(os.getcwd(), "chroma_db")
        
        # Check if directory exists
        if not os.path.exists(persist_directory):
            logger.warning(f"Chroma DB directory does not exist: {persist_directory}")
            return None
        
        # Try to load the client
        try:
            from langchain_nomic.embeddings import NomicEmbeddings
            embedding_model = NomicEmbeddings(model="nomic-embed-text-v1")
        except ImportError:
            # Fall back to a different embedding model if nomic is not available
            from langchain_huggingface import HuggingFaceEmbeddings
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        # Load the DB
        client = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        
        return client
        
    except Exception as e:
        logger.exception("Error loading Chroma client")
        return None

def test_chroma_query(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Test a query against the Chroma DB to verify it's working.
    
    Args:
        query: Query string to search for
        k: Number of results to return
        
    Returns:
        Dict containing the results of the query
    """
    result = {
        "success": False,
        "query": query,
        "results": [],
        "count": 0,
        "error": None
    }
    
    try:
        client = load_chroma_client()
        
        if client is None:
            result["error"] = "Failed to load Chroma client"
            return result
        
        # Query the DB
        retriever = client.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k}
        )
        
        docs = retriever.get_relevant_documents(query)
        
        # Format results
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            })
        
        result["results"] = results
        result["count"] = len(docs)
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
        logger.exception("Error in test_chroma_query")
    
    return result