# drive_pipeline.py
import os
import base64
import tempfile
from io import BytesIO
from typing import Any, Dict, List, Tuple
import os
import shutil 
import torch
from PIL import Image as PILImage
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from transformers import BlipProcessor, BlipForConditionalGeneration
from unstructured.partition.auto import partition
from unstructured.documents.elements import Image as UnstructuredImage, Table as UnstructuredTable, Text, Title, ListItem, NarrativeText, Element
from langchain.schema import Document
from langchain.text_splitter import (CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter, HTMLHeaderTextSplitter)
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.vectorstores import Chroma
import logging


# Ensure the logging setup is in place
logger = logging.getLogger(__name__)

# === API KEYS ===
os.environ["ATLAS_API_KEY"] = "nk-rjqDQKJtuoRaTcocIvaSJr6g5JItcyLvNJR4O7h153o"
embedding_model = NomicEmbeddings(model="nomic-embed-text-v1")

TEXT_TYPES = (Text, Title, ListItem, NarrativeText)

# --- Auth Service ---
def authenticate_gdrive():
    gauth = GoogleAuth()
    gauth.settings['get_refresh_token'] = False
    gauth.settings['oauth_scope'] = ["https://www.googleapis.com/auth/drive"]
    gauth.settings['service_config'] = {
        "client_json_file_path": "app/service_account.json",
        "client_user_email": "grademate-service@grademate-458004.iam.gserviceaccount.com",
    }
    gauth.ServiceAuth()
    return GoogleDrive(gauth)

# --- Fallback Extraction ---
def extract_text_from_pdf(path):
    from PyPDF2 import PdfReader
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""

def handle_fallback_file(path):
    """
    Fallback method for file extraction when partition fails or is unavailable
    """
    from .file_utils import handle_unsupported_file, get_supported_extension
    
    ext = os.path.splitext(path)[1].lower()
    file_type = get_supported_extension(path)
    
    try:
        if file_type == 'pdf':
            # For PDFs, use PyPDF2 as fallback
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return [NarrativeText(text=full_text)]
            
        else:
            # For other file types, use our specialized handlers
            return handle_unsupported_file(path)
            
    except Exception as e:
        # If all else fails, at least return something
        return [NarrativeText(text=f"Error processing file {os.path.basename(path)}: {str(e)}")]


# --- Image Extraction and Saving ---
def save_images_from_partition(documents, output_dir="saved_images"):
    """
    Extract and save each UnstructuredImage element's Base64 payload into the specified
    output directory as individual image files.
    
    Args:
        documents: List of dictionaries containing document elements
        output_dir: Directory to save extracted images
    
    Returns:
        int: Number of images saved
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    image_count = 0

    # Iterate through the documents and extract images
    for doc in documents:
        for element in doc.get("elements", []):
            if isinstance(element, UnstructuredImage):
                # Extract Base64 image data from metadata
                md = element.to_dict().get("metadata", {})
                b64 = md.get("image_base64", "")
                if not b64:
                    continue  # Skip if no image data is found
                
                # Log the image extraction
                logger.debug(f"Base64 data found for image: {element.id}")

                # Remove data URI prefix if present
                if "," in b64:
                    b64 = b64.split(",", 1)[1]

                try:
                    # Decode the base64 image data
                    raw = base64.b64decode(b64)
                    buffer = BytesIO(raw)
                    buffer.seek(0)

                    # Convert to a PIL Image and save
                    img = Image.open(buffer).convert("RGB")
                    filename = f"image_{element.id}_{image_count}.png"
                    filepath = os.path.join(output_dir, filename)
                    img.save(filepath)

                    image_count += 1
                    logger.debug(f"Saved image: {filepath}")

                except Exception as e:
                    logger.error(f"Error saving image {element.id}: {e}")

    return image_count


# --- Main Document Processing Logic ---
def load_and_partition_drive_documents(drive_id: str, download_dir: str, figures_base_dir: str, progress_callback=None) -> List[Dict]:
    """
    Load and partition documents from a Google Drive folder or file.
    Extract images and save them into the specified directory.

    Args:
        drive_id (str): The ID of the Google Drive folder or file.
        download_dir (str): Directory to temporarily download files.
        figures_base_dir (str): Base directory to save extracted figures/images.

    Returns:
        List[Dict]: A list of dictionaries containing the filename, extracted elements, and figures directory.
    """
    processed_docs = []
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(figures_base_dir, exist_ok=True)
    local_path = None

    drive = authenticate_gdrive()

    try:
        file = drive.CreateFile({'id': drive_id})
        file_title = file['title']
        local_path = os.path.join(download_dir, file_title)

        if progress_callback:
            progress_callback(f"\U0001F4E5 Downloading {file_title} from Drive...")

        file.GetContentFile(local_path)

        output_figures_path = os.path.join(figures_base_dir, os.path.splitext(file_title)[0])
        os.makedirs(output_figures_path, exist_ok=True)

        try:
            elements = partition(
                filename=local_path,
                strategy="hi_res",
                languages=["eng"],
                extract_image_block_to_payload=True,
                extract_image_block_types=["Image", "Table"],
                infer_table_structure=True,
                image_output_dir_path=output_figures_path
            )
            
            if elements is None or len(elements) == 0:
                # If partition returns nothing, fallback manually
                elements = handle_fallback_file(local_path)

        except Exception as e:
            # If partition itself raises error, fallback manually
            elements = handle_fallback_file(local_path)

        processed_docs.append({
            "filename": file_title,
            "elements": elements,
            "extracted_figures_dir": output_figures_path
        })

        # Extract and save images
        if progress_callback:
            progress_callback(f"🖼️ Extracting and saving images for {file_title}...")

        save_images_from_partition(processed_docs, output_dir=output_figures_path)

        if progress_callback:
            progress_callback(f"✅ Partitioned {file_title} with {len(elements)} elements, images saved.")

    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Error processing {drive_id}: {str(e)}")

    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)

    return processed_docs



# def save_unstructured_images(documents, output_dir="saved_images", progress_callback=None):
#     """
#     Decode and save each UnstructuredImage element's Base64 payload
#     into the specified output directory as individual image files.
    
#     Args:
#         documents: List of dictionaries containing document elements
#         output_dir: Directory to save extracted images
#         progress_callback: Optional callback function to report progress
    
#     Returns:
#         int: Number of images saved
#     """
#     # Create output directory if missing
#     os.makedirs(output_dir, exist_ok=True)
    
#     if progress_callback:
#         progress_callback(f"Creating image output directory: {output_dir}")

#     image_count = 0
#     for doc in documents:
#         for element in doc.get("elements", []):
#             if isinstance(element, UnstructuredImage):
#                 # Extract Base64 payload from metadata
#                 md = element.to_dict().get("metadata", {})
#                 b64 = md.get("image_base64", "")
#                 if not b64:
#                     continue
                
#                 # Strip off data URI header if present
#                 if "," in b64:
#                     b64 = b64.split(",", 1)[1]
                
#                 try:
#                     # Decode Base64 to bytes
#                     raw = base64.b64decode(b64)
                    
#                     # Wrap in BytesIO buffer
#                     buffer = BytesIO(raw)
#                     buffer.seek(0)
                    
#                     # Load as PIL Image
#                     img = PILImage.open(buffer).convert("RGB")
                    
#                     # Choose a filename and save
#                     elem_id = md.get("element_id") or getattr(element, "element_id", None) or element.id
#                     filename = f"image_{elem_id}_{image_count}.png"
#                     filepath = os.path.join(output_dir, filename)
#                     img.save(filepath)
                    
#                     image_count += 1
                    
#                     if progress_callback and image_count % 5 == 0:
#                         progress_callback(f"Saved {image_count} images so far...")
                    
#                 except Exception as e:
#                     if progress_callback:
#                         progress_callback(f"Error saving image: {str(e)}")
    
#     if progress_callback:
#         progress_callback(f"✅ Saved {image_count} images to {output_dir}")
    
#     return image_count




def setup_multi_query_retriever(vector_store, llm=None, search_kwargs=None):
    """
    Setup a MultiQueryRetriever for improved document retrieval.
    This generates multiple query variations to overcome limitations
    of single-query similarity search.
    
    Args:
        vector_store: Chroma or other vector store
        llm: Language model for query expansion (optional)
        search_kwargs: Arguments for the search method
    
    Returns:
        A retriever object (MultiQueryRetriever if llm provided, otherwise base retriever)
    """
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain_core.output_parsers import LineListOutputParser
    from langchain_core.prompts import PromptTemplate
    
    # Default search parameters
    search_kwargs = search_kwargs or {"fetch_k": 20, "k": 5}
    
    # Setup base retriever with MMR (Maximum Marginal Relevance)
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )
    
    # If no LLM provided, just return the base retriever
    if llm is None:
        return base_retriever
    
    # Setup output parser to convert LLM output to list of strings
    output_parser = LineListOutputParser()
    
    # Setup prompt template for query expansion
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    
    # Connect the prompt, llm and parser into a chain
    llm_chain = query_prompt | llm | output_parser
    
    # Create and return the MultiQueryRetriever
    return MultiQueryRetriever(
        retriever=base_retriever,
        llm_chain=llm_chain,
        parser_key="lines"
    )

def search_documents(vector_store, query, llm=None, k=5):
    """
    Search for documents relevant to the query.
    If an LLM is provided, uses MultiQueryRetriever for better results.
    
    Args:
        vector_store: Chroma or other vector store
        query: The search query
        llm: Language model for query expansion (optional)
        k: Number of documents to return
    
    Returns:
        List of retrieved documents
    """
    # Setup the retriever (multi-query if llm provided)
    retriever = setup_multi_query_retriever(
        vector_store, 
        llm=llm,
        search_kwargs={"fetch_k": k*4, "k": k}
    )
    
    # Perform the search
    results = retriever.invoke(query)
    
    return results


# --- Caption Generator ---
#  generate_captions_from_memory function

def generate_captions_from_memory(documents: List[Dict[str, Any]], model_name="Salesforce/blip-image-captioning-base", device=None, progress_callback=None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Generate captions for images and tables found in documents.
    Falls back to default captions if model loading or inference fails.
    
    Args:
        documents: List of document dictionaries with elements
        model_name: HuggingFace model name for image captioning
        device: Compute device (cuda or cpu)
        progress_callback: Optional callback function to report progress
        
    Returns:
        Tuple of (image_captions, table_captions) dictionaries
    """
    import logging
    logger = logging.getLogger(__name__)
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_captions, table_captions = {}, {}
    total = sum(len(doc.get("elements", [])) for doc in documents)
    
    if progress_callback:
        progress_callback(f"Generating captions for {total} elements...")
    
    # First, try to load the BLIP model for high-quality captions
    model = None
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained(model_name)
        # Try loading with explicit TensorFlow flag
        model = BlipForConditionalGeneration.from_pretrained(
            model_name, 
            from_tf=True,
            local_files_only=False
        ).to(device)
        if progress_callback:
            progress_callback(f"✅ BLIP model loaded successfully")
            
    except Exception as e:
        logger.warning(f"Could not load BLIP model: {str(e)}")
        if progress_callback:
            progress_callback(f"⚠️ Could not load BLIP model - using fallback captions")
    
    # Process documents - either with model or with fallback
    for doc_idx, doc in enumerate(documents):
        if progress_callback and doc_idx % max(1, len(documents)//5) == 0:
            progress_callback(f"Processing document {doc_idx+1}/{len(documents)}")
            
        for el in doc.get("elements", []):
            if isinstance(el, (UnstructuredImage, UnstructuredTable)):
                # Default/fallback captions
                caption = ""
                
                if isinstance(el, UnstructuredImage):
                    caption = "Image extracted from document"
                else:  # Table
                    caption = "Table extracted from document"
                
                # Try to generate a better caption if model loaded successfully
                if model is not None:
                    try:
                        md = el.to_dict().get("metadata", {})
                        b64 = md.get("image_base64", "")
                        
                        if b64:
                            raw = base64.b64decode(b64.split(",")[-1])
                            img = PILImage.open(BytesIO(raw)).convert("RGB")
                            inputs = processor(images=img, return_tensors="pt").to(device)
                            out = model.generate(**inputs)
                            ai_caption = processor.decode(out[0], skip_special_tokens=True)
                            
                            if ai_caption and len(ai_caption) > 5:
                                caption = ai_caption
                    except Exception as e:
                        logger.debug(f"Caption generation failed for element: {str(e)}")
                        # Fallback to default caption already set
                
                # Store the caption with the element ID
                elem_id = el.id
                if isinstance(el, UnstructuredImage):
                    image_captions[elem_id] = caption
                else:
                    table_captions[elem_id] = caption
    
    if progress_callback:
        progress_callback(f"✅ Generated {len(image_captions)} image captions and {len(table_captions)} table captions")
        
    return image_captions, table_captions

# --- Restructuring Elements ---
def restructure_all_elements_flat(documents: List[Dict[str, Any]]) -> Tuple[List[UnstructuredImage], List[UnstructuredTable], List[Element]]:
    images, tables, texts = [], [], []
    for doc in documents:
        for el in doc.get("elements", []):
            if isinstance(el, UnstructuredImage):
                images.append(el)
            elif isinstance(el, UnstructuredTable):
                tables.append(el)
            elif isinstance(el, TEXT_TYPES):
                texts.append(el)
    return images, tables, texts

# --- LangChain Documents Conversion ---
def convert_elements_to_langchain_docs(texts, images, tables, image_captions, table_captions) -> List[Document]:
    docs = []
    for el in texts:
        docs.append(Document(page_content=el.text, metadata={"type": "text", "element_id": el.id}))
    for el in images:
        cap = image_captions.get(el.id, "")
        if cap:
            docs.append(Document(page_content=f"[IMAGE:{el.id}] {cap}", metadata={"type": "image_caption", "element_id": el.id}))
    for el in tables:
        cap = table_captions.get(el.id, "")
        if cap:
            docs.append(Document(page_content=f"[TABLE:{el.id}] {cap}", metadata={"type": "table_caption", "element_id": el.id}))
    return docs

# --- Chunking Documents ---
def get_splitter_for_type(doc_type: str):
    if doc_type == "markdown":
        return MarkdownTextSplitter(chunk_size=512, chunk_overlap=64)
    if doc_type == "html":
        return HTMLHeaderTextSplitter(headers_to_split_on=["h1", "h2", "h3"], chunk_size=800, chunk_overlap=100)
    if doc_type == "json":
        return RecursiveJsonSplitter(max_chunk_size=500, min_chunk_size=100)
    if doc_type == "code":
        return RecursiveCharacterTextSplitter.from_language("python", chunk_size=1000, chunk_overlap=200)
    return CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def dynamic_chunk_documents(lc_docs: List[Document]) -> List[Document]:
    chunks = []
    for doc in lc_docs:
        splitter = get_splitter_for_type(doc.metadata.get("doc_type", "text"))
        chunks.extend(splitter.split_documents([doc]))
    return chunks

# --- Chroma Ingestion ---
def ingest_chroma(chunked_docs, embedding_model):
    """
    Create a new Chroma vector store with the given documents and embedding model.
    Will forcibly delete any existing Chroma DB at the specified location.
    
    Args:
        chunked_docs: List of document chunks to embed and store
        embedding_model: Embedding model to use for vectorization
        
    Returns:
        Chroma: A new Chroma vector store instance
    """
    import os
    import shutil
    import logging
    
    # Define the directory to persist the Chroma database
    persist_directory = os.path.join(os.getcwd(), "chroma_db")
    
    # Log what we're doing
    logger = logging.getLogger(__name__)
    logger.info(f"Setting up Chroma DB at {persist_directory}")
    
    # Force recreation by deleting existing DB if it exists
    if os.path.exists(persist_directory):
        logger.info(f"Removing existing Chroma DB at {persist_directory}")
        try:
            # Use rmtree to remove the directory and all its contents
            shutil.rmtree(persist_directory)
            logger.info(f"Successfully deleted existing Chroma DB")
        except Exception as e:
            logger.error(f"Error deleting Chroma DB: {str(e)}")
            # Continue anyway - Chroma might handle this
    
    # Ensure the directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # Log document stats
    logger.info(f"Ingesting {len(chunked_docs)} document chunks into Chroma")
    
    # Create the Chroma vector store from documents
    try:
        from langchain_community.vectorstores import Chroma
        vector_store = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        
        # Explicitly persist the database
        vector_store.persist()
        logger.info(f"Successfully created and persisted Chroma DB with {len(chunked_docs)} chunks")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating Chroma DB: {str(e)}")
        raise

# --- Full Main Runner ---
def main(file_ids: List[str], progress_callback=None):
    download_dir = "downloaded_files"
    figures_dir = "extracted_figures"
    images_dir = "saved_images"
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    
    download_dir = "downloaded_files"
    figures_dir = "extracted_figures"
    images_dir = "saved_images"
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    all_docs = []
    for drive_id in file_ids:
        if progress_callback:
            progress_callback(f"\U0001F4E5 Processing: {drive_id}")
        docs = load_and_partition_drive_documents(drive_id, download_dir, figures_dir, progress_callback)
        all_docs.extend(docs)

    if not all_docs:
        if progress_callback:
            progress_callback("❌ No valid documents.")
        return

    # Extract and save images from documents
    if progress_callback:
        progress_callback("🖼️ Extracting and saving images...")
    saved_image_count = save_unstructured_images(all_docs, output_dir=images_dir, progress_callback=progress_callback)

    images, tables, texts = restructure_all_elements_flat(all_docs)
    image_captions, table_captions = generate_captions_from_memory(all_docs, progress_callback=progress_callback)
    lc_docs = convert_elements_to_langchain_docs(texts, images, tables, image_captions, table_captions)
    chunks = dynamic_chunk_documents(lc_docs)

    if progress_callback:
        progress_callback("\U0001F4E2 Ingesting chunks into Chroma DB...")

    vector_store = ingest_chroma(chunks, embedding_model)

    # Example of how you might use the MultiQueryRetriever
    # from langchain_google_genai import ChatGoogleGenerativeAI
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    # example_result = search_documents(vector_store, "What is this document about?", llm=llm)
    
    if progress_callback:
        progress_callback("✅ Pipeline complete.")
        progress_callback(f"📊 Summary: Processed {len(all_docs)} documents, extracted {saved_image_count} images, created {len(chunks)} text chunks")
    
    return vector_store