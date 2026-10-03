"""Simple PDF extraction API for a medical-document RAG project.

This starter API accepts a PDF file and returns extracted text with page metadata.
It works best with PDFs that contain selectable text. Scanned PDFs may require OCR.
"""

from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from pypdf import PdfReader

app = FastAPI(title="Medical Document Extraction API")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Medical Document Extraction API is running"}


@app.post("/extract")
async def extract_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()

    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": page_number, "text": text})

    full_text = "\n\n".join(page["text"] for page in pages)

    return {
        "filename": file.filename,
        "page_count": len(pages),
        "text": full_text,
        "pages": pages,
    }
