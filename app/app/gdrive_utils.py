# gdrive_utils.py
from .models import DrivePDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .drive_pipeline import authenticate_gdrive  #  Import your OAuth login
import tempfile
import os
from PyPDF2 import PdfReader

def chunk_pdf_content(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    return splitter.split_text(text)


DOWNLOAD_DIR = "downloaded_files"

def fetch_drive_pdfs_and_chunk(user, logs):
    folder_id = "1LyiyENXg85q-uLQjlddNEeRsi6bgKdTq"
    logs.append(" Fetching public folder using PyDrive2...")

    try:
        drive = authenticate_gdrive()
        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

        if not file_list:
            logs.append(" No files found.")
            return []

        logs.append(f" Found {len(file_list)} files.")

        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

        for file in file_list:
            fname = file['title']
            file_id = file['id']   #  correct

            if not fname.lower().endswith((".pdf", ".docx", ".txt", ".md", ".csv", ".html", ".py", ".ipynb", ".json", ".xml")):
                logs.append(f"⏭ Skipping unsupported file: {fname}")
                continue

            full_path = os.path.join(DOWNLOAD_DIR, fname)
            logs.append(f"📥 Downloading {fname}...")

            try:
                file.GetContentFile(full_path)
                logs.append(f" Downloaded {fname} into {full_path}.")

                # IMPORTANT: Save correct file_id (not filename)
                if not DrivePDF.objects.filter(user=user, title=fname).exists():
                    DrivePDF.objects.create(
                        user=user,
                        title=fname,
                        drive_file_id=file_id  #  Save real ID
                    )
                    logs.append(f"✅ Registered {fname} into database.")
                else:
                    logs.append(f"ℹ️ {fname} already registered, skipping.")

            except Exception as e:
                logs.append(f" Failed to download/register {fname}: {str(e)}")

        return list(DrivePDF.objects.filter(user=user).order_by('-uploaded_at')[:10])

    except Exception as e:
        logs.append(f" Fetch failed: {str(e)}")
        raise e
