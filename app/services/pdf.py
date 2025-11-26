import io
import os
import tempfile
from typing import List, Tuple
from datetime import datetime
import aiohttp
from PyPDF2 import PdfReader
from app.services.rag import get_text_chunks, get_vector_store
from app.core.config import settings

async def download_file_from_url(url: str, timeout: int = 300) -> bytes:
    """
        File content as bytes
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            response.raise_for_status()
            return await response.read()


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:

    text = ""
    try:
        with io.BytesIO(pdf_bytes) as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text


def save_temp_file(content: bytes, suffix: str = ".pdf") -> str:
    """
    Save content to a temporary file and return the path.
    
    Args:
        content: File content as bytes
        suffix: File extension
        
    Returns:
        Path to temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp_file.write(content)
        temp_file.flush()
        return temp_file.name
    finally:
        temp_file.close()


def delete_temp_file(file_path: str):
    """
    Safely delete a temporary file.
    
    Args:
        file_path: Path to the temporary file
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting temp file {file_path}: {e}")


async def process_files_from_urls(download_urls: List[Tuple[str, str]], db) -> int:
    """
    Download files from URLs, extract text, create embeddings, and clean up.
    
    Args:
        download_urls: List of tuples (download_url, url_hash)
        db: MongoDB database instance
        
    Returns:
        Number of successfully processed files
    """
    
    processed_count = 0
    all_text_chunks = []
    
    for url, url_hash in download_urls:
        temp_file_path = None
        try:
            # Check if already processed (防止重複)
            if db.processed_files.find_one({"url_hash": url_hash}):
                print(f"File with hash {url_hash} already processed, skipping...")
                continue
            
            # Download file
            print(f"Downloading file from {url}...")
            file_content = await download_file_from_url(url)
            
            # Extract text
            text = extract_text_from_pdf_bytes(file_content)
            
            if not text.strip():
                print(f"No text extracted from {url}, skipping...")
                continue
            
            # Create text chunks
            chunks = get_text_chunks(text, settings.MODEL_NAME)
            all_text_chunks.extend(chunks)
            
            # Mark as processed
            db.processed_files.insert_one({
                "url_hash": url_hash,
                "processed_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "chunks_count": len(chunks)
            })
            
            # Update files collection
            db.files.update_one(
                {"url_hash": url_hash},
                {"$set": {
                    "embedding_created": True,
                    "processed_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }}
            )
            
            processed_count += 1
            print(f"Successfully processed file from {url}")
            
        except Exception as e:
            print(f"Error processing file from {url}: {e}")
        finally:
            # Clean up temporary file if created
            if temp_file_path:
                delete_temp_file(temp_file_path)
    
    # Create/update vector store with all chunks
    if all_text_chunks:
        print(f"Creating vector store with {len(all_text_chunks)} chunks...")
        get_vector_store(all_text_chunks, settings.MODEL_NAME, settings.API_KEY)
    
    return processed_count


def read_all_pdfs_text(db) -> str:
    """
    Legacy function - kept for backward compatibility.
    Note: This should not be used in the new flow.
    """
    text = ""
    files = list(db.files.find())
    if not files:
        return text

    # This is deprecated - files are no longer stored in GridFS
    # Instead, use process_files_from_urls
    return text
