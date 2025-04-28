# Requirements:
!pip install langchain unstructured[all-docs] pydantic lxml openai chromadb tiktoken pytesseract langchain_google_genai
!pip install langchain-huggingface transformers torch
!pip install -U langchain-community
!pip install pytesseract

!apt-get install -y poppler-utils
!apt-get install -y tesseract-ocr

!pip install google-generativeai

!pip install langchain-google-community[drive]
!pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib


import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from unstructured.partition.auto import partition
import traceback # Import traceback for more detailed error info (optional)




# --------- SETTINGS ---------
SERVICE_ACCOUNT_FILE = "/content/steady-citron-457407-q7-5c09ed11e0d4.json"



# --------- AUTH ---------
try:
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    drive_service = build('drive', 'v3', credentials=creds)
    print("Google Drive authentication successful.")
except Exception as e:
    print(f"Error during Google Drive authentication: {e}")
    print("Please check your service account file path and ensure it's valid.")
    # In a real application, you might want to exit or handle this differently
    # For this script, we'll let it continue but no files will be loaded.
    drive_service = None

def load_and_partition_drive_documents(drive_id, download_dir, figures_base_dir):
    """
    Loads and partitions documents from a Google Drive folder or file.

    Args:
        drive_id (str): The ID of the Google Drive folder or file.
        download_dir (str): Directory to temporarily download files.
        figures_base_dir (str): Base directory to save extracted figures/images.

    Returns:
        List[Dict]: A list of dictionaries, each containing filename and extracted elements.
    """
    processed_docs = []

    try:
        # Retrieve metadata to determine if the ID is a folder or file
        file_metadata = drive_service.files().get(fileId=drive_id, fields="id, name, mimeType").execute()
        mime_type = file_metadata.get("mimeType")
        name = file_metadata.get("name")

        if mime_type == "application/vnd.google-apps.folder":
            # It's a folder; fetch all files within
            files = get_files_from_folder(drive_id)
        else:
            # It's a single file
            files = [file_metadata]

    except Exception as e:
        print(f"Error retrieving metadata for ID {drive_id}: {e}")
        return []

    if not files:
        print("No files to process.")
        return []

    for file in files:
        file_id = file["id"]
        name = file["name"]

        local_path = os.path.join(download_dir, name)
        output_figures_path = os.path.join(figures_base_dir, os.path.splitext(name)[0])
        os.makedirs(output_figures_path, exist_ok=True)

        print(f"\nAttempting to process: {name}")

        try:
            request = drive_service.files().get_media(fileId=file_id)

            # Download file to temporary directory
            with io.FileIO(local_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status = downloader.next_chunk()
                    _, done = status

            print(f"Downloaded: {name} to {local_path}")

            # Check if the file is a ZIP archive
            if name.lower().endswith('.zip'):
                # Extract ZIP file
                with zipfile.ZipFile(local_path, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
                os.remove(local_path)  # Remove ZIP file after extraction

                # Process each extracted file
                for root, _, filenames in os.walk(download_dir):
                    for filename in filenames:
                        extracted_path = os.path.join(root, filename)
                        try:
                            elements = partition(
                                filename=extracted_path,
                                strategy="hi_res",
                                languages=["eng"],
                                extract_image_block_to_payload=True,
                                extract_image_block_types=["Image", "Table"],
                                infer_table_structure=True,
                                image_output_dir_path=output_figures_path
                            )
                            processed_docs.append({
                                "filename": filename,
                                "elements": elements,
                                "extracted_figures_dir": output_figures_path
                            })
                            print(f"✅ Successfully loaded and partitioned: {filename} ({len(elements)} elements found)")
                        except Exception as e:
                            print(f"❌ Failed to partition {filename}: {e}")
                        finally:
                            os.remove(extracted_path)  # Clean up extracted file
            else:
                # Process non-ZIP file
                elements = partition(
                    filename=local_path,
                    strategy="hi_res",
                    languages=["eng"],
                    extract_image_block_to_payload=True,
                    extract_image_block_types=["Image", "Table"],
                    infer_table_structure=True,
                    image_output_dir_path=output_figures_path
                )
                processed_docs.append({
                    "filename": name,
                    "elements": elements,
                    "extracted_figures_dir": output_figures_path
                })
                print(f"✅ Successfully loaded and partitioned: {name} ({len(elements)} elements found)")

        except Exception as e:
            print(f"❌ Failed to process {name}: {e}")
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

    return processed_docs


import os
import base64
from io import BytesIO
from PIL import Image as PILImage
from unstructured.documents.elements import Image as UnstructuredImage

def save_unstructured_images(documents, output_dir="saved_images"):
    """
    Decode and save each UnstructuredImage element's Base64 payload
    into the specified output directory as individual image files.
    """
    # 1. Create output directory (and any parents) if missing
    os.makedirs(output_dir, exist_ok=True)

    image_count = 0
    for doc in documents:
        for element in doc.get("elements", []):
            if isinstance(element, UnstructuredImage):
                # 2. Extract Base64 payload from metadata
                data = element.to_dict().get("metadata", {}).get("image_base64", "")
                if not data:
                    continue
                # 3. Strip off data URI header if present
                if "," in data:
                    data = data.split(",", 1)[1]
                # 4. Decode Base64 to bytes
                raw_bytes = base64.b64decode(data)
                # 5. Wrap in BytesIO buffer
                buffer = BytesIO(raw_bytes)
                buffer.seek(0)
                # 6. Load as PIL Image
                pil_img = PILImage.open(buffer)
                pil_img.load()  # optional: ensure full read

                # 7. Choose a filename and save
                filename = f"image_{image_count}.png"
                filepath = os.path.join(output_dir, filename)
                pil_img.save(filepath)

                print(f"Saved image #{image_count} to {filepath}")
                image_count += 1

# Example usage:





import base64
from io import BytesIO
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image as PILImage
from transformers import BlipProcessor, BlipForConditionalGeneration
from unstructured.documents.elements import Image as UnstructuredImage, Table as UnstructuredTable

def generate_captions_from_memory(
    documents: List[Dict[str, Any]],
    model_name: str = "Salesforce/blip-image-captioning-base",
    device: str = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Iterate all Image and Table elements in `documents`, decode their in‑payload Base64,
    and generate captions via BLIP (in memory, no disk I/O).

    Args:
        documents: list of dicts, each with key "elements": List[Element]
        model_name: HF model to load
        device: 'cuda' or 'cpu' (auto‑chosen if None)

    Returns:
        (image_captions, table_captions):
            image_captions: Dict[element_id, caption]
            table_captions: Dict[element_id, caption]
    """
    # Device setup
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load BLIP processor & model once
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    image_captions: Dict[str, str] = {}
    table_captions: Dict[str, str] = {}
    img_count = 0
    tbl_count = 0

    for doc in documents:
        for el in doc.get("elements", []):
            # Only process images or tables
            if isinstance(el, UnstructuredImage) or isinstance(el, UnstructuredTable):
                # 1) Extract Base64 from metadata
                md = el.to_dict().get("metadata", {})
                b64 = md.get("image_base64", "")
                if not b64:
                    continue
                if "," in b64:
                    b64 = b64.split(",", 1)[1]

                # 2) Decode & load PIL image
                raw = base64.b64decode(b64)
                buf = BytesIO(raw)
                buf.seek(0)
                img = PILImage.open(buf).convert("RGB")

                # 3) Generate caption
                inputs = processor(images=img, return_tensors="pt").to(device)
                out = model.generate(**inputs)
                cap = processor.decode(out[0], skip_special_tokens=True)

                # 4) Determine key
                elem_id = md.get("element_id") or getattr(el, "element_id", None) or el.id
                if not elem_id:
                    # fallback to type+count
                    if isinstance(el, UnstructuredImage):
                        elem_id = f"image_{img_count}"
                        img_count += 1
                    else:
                        elem_id = f"table_{tbl_count}"
                        tbl_count += 1

                # 5) Store in respective dict
                if isinstance(el, UnstructuredImage):
                    image_captions[elem_id] = cap
                else:
                    table_captions[elem_id] = cap

    return image_captions, table_captions

# -----------------------
# Example usage:

# docs = partition(..., extract_image_block_to_payload=True, extract_image_block_types=["Image","Table"])

# print("Images:", img_caps)
# print("Tables:", tbl_caps)


#documents = load_and_partition_drive_documents(FOLDER_ID, DOWNLOAD_TMP_DIR, EXTRACTED_FIGURES_BASE_DIR)

from typing import Any, Dict, List, Tuple
from unstructured.documents.elements import (
    Element,
    Image as UnstructuredImage,
    Table as UnstructuredTable,
    Text,
    Title,
    ListItem,
    NarrativeText,
)

# Define which classes count as “text”
TEXT_TYPES = (Text, Title, ListItem, NarrativeText)

def restructure_all_elements_flat(
    documents: List[Dict[str, Any]]
) -> Tuple[List[UnstructuredImage], List[UnstructuredTable], List[Element]]:
    """
    Flattens a list of parsed documents into three lists:
      1) all UnstructuredImage elements
      2) all UnstructuredTable elements
      3) all text-based elements (Text, Title, ListItem, NarrativeText)

    Args:
        documents: List of dicts, each with keys:
            - "filename": str
            - "elements": List[Element]

    Returns:
        Tuple of three lists: (all_images, all_tables, all_texts)
    """
    all_images: List[UnstructuredImage] = []
    all_tables: List[UnstructuredTable] = []
    all_texts: List[Element] = []

    for doc in documents:
        for el in doc.get("elements", []):
            if isinstance(el, UnstructuredImage):
                all_images.append(el)
            elif isinstance(el, UnstructuredTable):
                all_tables.append(el)
            elif isinstance(el, TEXT_TYPES):
                all_texts.append(el)
            # else: ignore other element types

    return all_images, all_tables, all_texts

# --------------------
# Example usage:

# Suppose you already have:
# documents = partition(..., extract_image_block_to_payload=True, extract_image_block_types=["Image","Table"])

# You can now pass these into your captioning or LangChain conversion steps.






#!pip install langchain-text-splitters




from typing import List, Dict, Any
from langchain.schema import Document
from unstructured.documents.elements import (
    Text,
    Title,
    ListItem,
    NarrativeText,
    Image as UnstructuredImage,
    Table as UnstructuredTable,
)

TEXT_TYPES = (Text, Title, ListItem, NarrativeText)

def convert_elements_to_langchain_docs(
    texts: List[Any],
    images: List[UnstructuredImage],
    tables: List[UnstructuredTable],
    image_captions: Dict[str, str],
    table_captions: Dict[str, str],
) -> List[Document]:
    """
    Build a unified list of LangChain Documents from text elements,
    image elements + captions, and table elements + captions.
    """
    docs: List[Document] = []

    # 1) Text elements → Documents
    for el in texts:
        docs.append(Document(
            page_content=el.text,
            metadata={
                "type": "text",
                "element_id": el.id,
                "source": getattr(el.metadata, "filename", None),
            },
        ))

    # 2) Image elements → caption Docs
    for el in images:
        el_dict = el.to_dict()
        elem_id = el_dict.get("id") or el.id
        caption = image_captions.get(elem_id, "")
        # skip if no caption
        if not caption:
            continue
        base64_str = el_dict.get("metadata", {}).get("image_base64")
        docs.append(Document(
            page_content=f"[IMAGE:{elem_id}] {caption}",
            metadata={
                "type": "image_caption",
                "element_id": elem_id,
                "image_base64": base64_str,
            },
        ))

    # 3) Table elements → caption Docs
    for el in tables:
        el_dict = el.to_dict()
        elem_id = el_dict.get("id") or el.id
        caption = table_captions.get(elem_id, "")
        if not caption:
            continue
        base64_str = el_dict.get("metadata", {}).get("image_base64")
        docs.append(Document(
            page_content=f"[TABLE:{elem_id}] {caption}",
            metadata={
                "type": "table_caption",
                "element_id": elem_id,
                "image_base64": base64_str,
            },
        ))

    return docs

# ----------------------
# EXAMPLE PIPELINE USAGE:

# 1) Flatten raw elements:
# all_images, all_tables, all_texts = restructure_all_elements_flat(documents)

# 2) Generate captions:
# img_caps, tbl_caps = generate_captions_from_memory(documents)

# 3) Convert to LangChain Documents:


# 4) Ready for dynamic_chunk_documents(lc_docs) or embedding!


from typing import List
from langchain.schema import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    HTMLHeaderTextSplitter,
)
from langchain_text_splitters import RecursiveJsonSplitter


def get_splitter_for_type(doc_type: str):
    """
    Return the optimal TextSplitter for the given document type.
    """
    if doc_type == "markdown":
        # Splits at Markdown headings and paragraphs for coherent sections
        return MarkdownTextSplitter(chunk_size=512, chunk_overlap=64)  # :contentReference[oaicite:6]{index=6}
    if doc_type == "html":
        # Splits by HTML header tags, preserving section context
        return HTMLHeaderTextSplitter(
            headers_to_split_on=["h1", "h2", "h3"],
            chunk_size=800,
            chunk_overlap=100,
        )  # :contentReference[oaicite:7]{index=7}
    if doc_type == "json":
        # Recursively splits nested JSON objects into character‑bounded chunks
        return RecursiveJsonSplitter(max_chunk_size=500, min_chunk_size=100)  # :contentReference[oaicite:8]{index=8}
    if doc_type == "code":
        # Language‑aware splitting using syntax separators for code
        return RecursiveCharacterTextSplitter.from_language(
            "python", chunk_size=1000, chunk_overlap=200
        )  # :contentReference[oaicite:9]{index=9}
    # Fallback for plain text: fixed‑size character chunks with overlap
    return CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # :contentReference[oaicite:10]{index=10}

def dynamic_chunk_documents(
    lc_docs: List[Document]
) -> List[Document]:
    """
    Apply document‑specific chunking to a list of LangChain Documents.

    Args:
        lc_docs: List of Documents, each with metadata["doc_type"] in
                 {"markdown", "html", "json", "code", "text"}.

    Returns:
        Flat list of smaller Document chunks.
    """
    all_chunks: List[Document] = []
    for doc in lc_docs:
        # Determine which splitter to use based on doc_type
        doc_type = doc.metadata.get("doc_type", "text").lower()
        splitter = get_splitter_for_type(doc_type)
        # Split the document into chunks
        chunks = splitter.split_documents([doc])  # :contentReference[oaicite:11]{index=11}
        # Annotate each chunk with its parent type for traceability
        for chunk in chunks:
            chunk.metadata["parent_doc_type"] = doc_type
            all_chunks.append(chunk)
    return all_chunks

# --------------------------
# Example usage:
#
# Assume `lc_docs` is your list of Documents already
# created from text, image captions, etc., each having
# metadata["doc_type"] ∈ {"markdown","html","json","code","text"}.
#

# print(f"Total chunks generated: {len(chunked_docs)}")
# for c in chunked_docs[:3]:
#     print(c.metadata["parent_doc_type"], "→", c.page_content[:100])


from langchain_community.vectorstores import Chroma

def ingest_chroma(chunked_docs,embedding_model):
# Define the directory to persist the Chroma database
    persist_directory = "./chroma_db"

    # Create the Chroma vector store from documents
    vector_store = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    return vector_store
# Persist the vector store to disk




import re
from urllib.parse import urlparse, parse_qs

def extract_drive_id(url: str) -> str | None:
    """
    Return the file/folder ID portion of any common Google-Drive link.

    Works with links like:
      • https://drive.google.com/file/d/<ID>/view?usp=sharing
      • https://drive.google.com/uc?id=<ID>&export=download
      • https://drive.google.com/open?id=<ID>
      • https://drive.google.com/drive/folders/<ID>?usp=drive_link

    Returns
    -------
    str | None
        The 33-character Drive ID if found, otherwise None.
    """
    # pattern 1:  .../d/<id>/...
    m = re.search(r"/d/([a-zA-Z0-9_-]{10,})", url)
    if m:
        return m.group(1)

    # pattern 2:  .../folders/<id>
    m = re.search(r"/folders?/([a-zA-Z0-9_-]{10,})", url)
    if m:
        return m.group(1)

    # pattern 3:  id=<id> in the query string
    qs_vals = parse_qs(urlparse(url).query).get("id")
    if qs_vals:
        return qs_vals[0]

    # nothing matched
    return None


from langchain_huggingface import HuggingFaceEmbeddings

# Specify the model name; you can choose any model from Hugging Face's model hub
model_name = "sentence-transformers/all-mpnet-base-v2"

# Optional: Define model and encoding parameters
model_kwargs = {'device': 'cpu'}  # or 'cuda' if using GPU
encode_kwargs = {'normalize_embeddings': True}

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


DOWNLOAD_TMP_DIR = "/content/drive_downloads"
EXTRACTED_FIGURES_BASE_DIR = os.path.join(os.getcwd(), "figures") # Base directory for saving figures

# Ensure download and figures base directories exist
os.makedirs(DOWNLOAD_TMP_DIR, exist_ok=True)
os.makedirs(EXTRACTED_FIGURES_BASE_DIR, exist_ok=True)



FOLDER_ID = "1LyiyENXg85q-uLQjlddNEeRsi6bgKdTq" # misc documents

link = "https://drive.google.com/file/d/1YtSduxc-jYJAimnslOgtNUWTBX_fa2By/view?usp=drive_link"
DRIVE_ID = extract_drive_id(link)

#documents = load_and_partition_drive_documents(FOLDER_ID, DOWNLOAD_TMP_DIR, EXTRACTED_FIGURES_BASE_DIR)
documents = load_and_partition_drive_documents(DRIVE_ID, DOWNLOAD_TMP_DIR, EXTRACTED_FIGURES_BASE_DIR)
print(f"\n🎉 Done processing files. Successfully partitioned {len(documents)} documents.")

print(f"Extracted figures and tables are saved in subdirectories within: {EXTRACTED_FIGURES_BASE_DIR}")



save_unstructured_images(documents, output_dir="my_extracted_images")
img_caps, tbl_caps = generate_captions_from_memory(documents)
all_images, all_tables, all_texts = restructure_all_elements_flat(documents)

print(f"Found {len(all_images)} images, {len(all_tables)} tables, {len(all_texts)} text elements.")

lc_docs = convert_elements_to_langchain_docs(
     texts=all_texts,
     images=all_images,
     tables=all_tables,
     image_captions=img_caps,
     table_captions=tbl_caps,
 )


chunked_docs = dynamic_chunk_documents(lc_docs)
vector_store = ingest_chroma(chunked_docs,embedding_model)
vector_store.persist()












#chunked_docs

import os
import base64
import zipfile
import tempfile
import shutil
from io import BytesIO
from pathlib import Path
from PIL import Image as PILImage
from unstructured.partition.auto import partition
from unstructured.documents.elements import Image as UnstructuredImage

def save_unstructured_images_from_directory(
    source_path: str,
    output_dir: str = "saved_images",
    strategy: str = "hi_res",
    languages: list[str] = ["eng"]
) -> list[dict]:
    """
    Walk a directory (or single file, or ZIP), partition each document,
    extract UnstructuredImage elements, decode & save them, and return
    a summary of processed docs.

    Args:
        source_path: Path to a directory, file, or ZIP archive.
        output_dir:  Base directory in which to save extracted images.
        strategy:    Unstructured partition strategy.
        languages:   List of language codes for OCR partitioning.

    Returns:
        A list of dicts, each with:
          {
            "source": <filename>,
            "elements": <List[Element]>,
            "saved_images": <List[path to saved image files]>
          }
    """
    processed = []
    source = Path(source_path)

    # Gather input files
    to_process = []
    if source.is_dir():
        # walk directory for files
        for path in source.rglob("*"):
            if path.is_file():
                to_process.append(path)
    elif zipfile.is_zipfile(source):
        # extract ZIP to temp dir
        tmpdir = Path(tempfile.mkdtemp(prefix="unzipped_"))
        with zipfile.ZipFile(source, "r") as zf:
            zf.extractall(tmpdir)
        for path in tmpdir.rglob("*"):
            if path.is_file():
                to_process.append(path)
    else:
        # single file
        to_process.append(source)

    # ensure base output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # process each file
    for file_path in to_process:
        name       = file_path.name
        stem       = file_path.stem
        elements   = partition(
            filename=str(file_path),
            strategy=strategy,
            languages=languages,
            extract_image_block_to_payload=True,
            extract_image_block_types=["Image"],
            infer_table_structure=False,
            image_output_dir_path=None
        )

        processed.append({
            "source": str(file_path),
            "elements": elements

        })

    # clean up tempdir if created
    if 'tmpdir' in locals():
        shutil.rmtree(tmpdir, ignore_errors=True)

    return processed


!mkdir my_docs_folder

documents = save_unstructured_images_from_directory(
    source_path="my_docs_folder",
    output_dir="extracted_images"
)
print(f"\n🎉 Done processing files. Successfully partitioned {len(documents)} documents.")




save_unstructured_images(documents, output_dir="my_extracted_images")
img_caps, tbl_caps = generate_captions_from_memory(documents)
all_images, all_tables, all_texts = restructure_all_elements_flat(documents)

print(f"Found {len(all_images)} images, {len(all_tables)} tables, {len(all_texts)} text elements.")

lc_docs = convert_elements_to_langchain_docs(
     texts=all_texts,
     images=all_images,
     tables=all_tables,
     image_captions=img_caps,
     table_captions=tbl_caps,
 )


chunked_docs = dynamic_chunk_documents(lc_docs)
vector_store = ingest_chroma(chunked_docs,embedding_model)
vector_store.persist()





#embeddings













# Define your query
query = "What is segmeentation "

# Perform the similarity search
similar_docs = vector_store.similarity_search(query, k=5)

# Display the results
for doc in similar_docs:
    print(doc.page_content)








# now test for the gdrive gENNERIC FODLER / FILE /ZIP FILE







def load_and_partition_drive_documents(drive_id, download_dir, figures_base_dir):
    """
    Loads and partitions documents from a Google Drive folder or file.

    Args:
        drive_id (str): The ID of the Google Drive folder or file.
        download_dir (str): Directory to temporarily download files.
        figures_base_dir (str): Base directory to save extracted figures/images.

    Returns:
        List[Dict]: A list of dictionaries, each containing filename and extracted elements.
    """
    processed_docs = []

    try:
        # Retrieve metadata to determine if the ID is a folder or file
        file_metadata = drive_service.files().get(fileId=drive_id, fields="id, name, mimeType").execute()
        mime_type = file_metadata.get("mimeType")
        name = file_metadata.get("name")

        if mime_type == "application/vnd.google-apps.folder":
            # It's a folder; fetch all files within
            files = get_files_from_folder(drive_id)
        else:
            # It's a single file
            files = [file_metadata]

    except Exception as e:
        print(f"Error retrieving metadata for ID {drive_id}: {e}")
        return []

    if not files:
        print("No files to process.")
        return []

    for file in files:
        file_id = file["id"]
        name = file["name"]

        local_path = os.path.join(download_dir, name)
        output_figures_path = os.path.join(figures_base_dir, os.path.splitext(name)[0])
        os.makedirs(output_figures_path, exist_ok=True)

        print(f"\nAttempting to process: {name}")

        try:
            request = drive_service.files().get_media(fileId=file_id)

            # Download file to temporary directory
            with io.FileIO(local_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status = downloader.next_chunk()
                    _, done = status

            print(f"Downloaded: {name} to {local_path}")

            # Check if the file is a ZIP archive
            if name.lower().endswith('.zip'):
                # Extract ZIP file
                with zipfile.ZipFile(local_path, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
                os.remove(local_path)  # Remove ZIP file after extraction

                # Process each extracted file
                for root, _, filenames in os.walk(download_dir):
                    for filename in filenames:
                        extracted_path = os.path.join(root, filename)
                        try:
                            elements = partition(
                                filename=extracted_path,
                                strategy="hi_res",
                                languages=["eng"],
                                extract_image_block_to_payload=True,
                                extract_image_block_types=["Image", "Table"],
                                infer_table_structure=True,
                                image_output_dir_path=output_figures_path
                            )
                            processed_docs.append({
                                "filename": filename,
                                "elements": elements,
                                "extracted_figures_dir": output_figures_path
                            })
                            print(f"✅ Successfully loaded and partitioned: {filename} ({len(elements)} elements found)")
                        except Exception as e:
                            print(f"❌ Failed to partition {filename}: {e}")
                        finally:
                            os.remove(extracted_path)  # Clean up extracted file
            else:
                # Process non-ZIP file
                elements = partition(
                    filename=local_path,
                    strategy="hi_res",
                    languages=["eng"],
                    extract_image_block_to_payload=True,
                    extract_image_block_types=["Image", "Table"],
                    infer_table_structure=True,
                    image_output_dir_path=output_figures_path
                )
                processed_docs.append({
                    "filename": name,
                    "elements": elements,
                    "extracted_figures_dir": output_figures_path
                })
                print(f"✅ Successfully loaded and partitioned: {name} ({len(elements)} elements found)")

        except Exception as e:
            print(f"❌ Failed to process {name}: {e}")
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

    return processed_docs


















































from langchain_google_genai import ChatGoogleGenerativeAI  # :contentReference[oaicite:0]{index=0}
from langchain.retrievers.multi_query import MultiQueryRetriever  # :contentReference[oaicite:1]{index=1}


from typing import List

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', google_api_key=GOOGLE_API_KEY) # best method

# Chain
llm_chain = QUERY_PROMPT | llm | output_parser

# Other inputs
question = "What are the approaches to Task Decomposition?"

base_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"fetch_k": 20, "k": 5}
)# Run


retriever = MultiQueryRetriever(
    retriever=base_retriever, llm_chain=llm_chain, parser_key="lines",

)  # "lines" is the key (attribute name) of the parsed output

# Results
unique_docs = retriever.invoke("What does the course say about regression?")
len(unique_docs)

for doc in unique_docs:
  print(doc.page_content)