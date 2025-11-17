import io
from typing import Iterable
from bson import ObjectId
from gridfs import GridFS
from PyPDF2 import PdfReader


def read_all_pdfs_text(db) -> str:
    text = ""
    files = list(db.files.find())
    if not files:
        return text

    fs = GridFS(db)
    for file in files:
        try:
            grid_file = fs.get(ObjectId(file["file_id"]))
            with io.BytesIO(grid_file.read()) as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception:
            continue
    return text
