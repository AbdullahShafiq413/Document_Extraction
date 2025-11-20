# main.py — tiny FastAPI app; heavy logic is imported lazily

import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="PDF Extractor (Gemini-only)",
    version="1.1.2",
)

# Open CORS for testing; tighten in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "gemini_key_present": bool(os.environ.get("GOOGLE_API_KEY"))}

def _coerce_max_pages(raw: Optional[str]) -> Optional[int]:
    """
    Accepts None / "" / "0" / invalid strings -> returns None (process all pages).
    Accepts positive integer strings -> returns int.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s == "0":
        return None
    try:
        val = int(s)
        if val <= 0:
            return None
        return val
    except Exception:
        return None

@app.post("/extract")
async def extract_endpoint(
    file: UploadFile = File(..., description="PDF file upload"),
    lang: str = Form("eng"),
    dpi: int = Form(300),
    ocr_psm: int = Form(3),
    force_ocr: bool = Form(False),
    ocr_on_empty_only: bool = Form(True),
    max_pages: Optional[str] = Form(None),  # accept string from form; coerce safely
    include_text_preview: bool = Form(False),
):
    # Lazy import to avoid any import-time errors killing 'app'
    try:
        from extractor import (
            extract_pdf_content,
            build_single_text,
            gemini_extract_fields_only,
            gemini_extract_factory_review,
        )
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to import extractor: {e}")

    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set (.env).")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    # Normalize max_pages (blank/invalid -> None)
    max_pages_int = _coerce_max_pages(max_pages)

    # Save upload to a temp file
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            tmp.write(content)
            tmp_path = tmp.name
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded PDF: {e}")

    try:
        # PDF -> text (native first, OCR fallback)
        pages = extract_pdf_content(
            tmp_path,
            dpi=dpi,
            lang=lang,
            ocr_psm=ocr_psm,
            ocr_on_empty_only=ocr_on_empty_only,
            force_ocr=force_ocr,
            max_pages=max_pages_int,
        )
        doc_text = build_single_text(pages)

        # Gemini-only extraction
        data = gemini_extract_fields_only(doc_text)

        if include_text_preview:
            return JSONResponse({
                "data": data,
                "preview": doc_text[:1000] + ("…[truncated]" if len(doc_text) > 1000 else ""),
                "meta": {
                    "pages": len(pages),
                    "used_ocr_pages": sum(1 for p in pages if p.get('used_ocr')),
                }
            })
        return JSONResponse(data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/extract-factory-review")
async def extract_factory_review_endpoint(
    file: UploadFile = File(..., description="PDF file upload"),
    lang: str = Form("eng"),
    dpi: int = Form(300),
    ocr_psm: int = Form(3),
    force_ocr: bool = Form(False),
    ocr_on_empty_only: bool = Form(True),
    max_pages: Optional[str] = Form(None),
    include_text_preview: bool = Form(False),
):
    try:
        from extractor import (
            extract_pdf_content,
            build_single_text,
            gemini_extract_factory_review,
            extract_audit_date  # Import the function for extracting audit date
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import extractor: {e}")

    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set (.env).")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    max_pages_int = _coerce_max_pages(max_pages)

    import tempfile
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            tmp.write(content)
            tmp_path = tmp.name
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded PDF: {e}")

    try:
        # PDF -> text
        pages = extract_pdf_content(
            tmp_path,
            dpi=dpi,
            lang=lang,
            ocr_psm=ocr_psm,
            ocr_on_empty_only=ocr_on_empty_only,
            force_ocr=force_ocr,
            max_pages=max_pages_int,
        )
        doc_text = build_single_text(pages)

        # Extract the Audit Date
        audit_date = extract_audit_date(doc_text)

        # Factory Review extraction
        data = gemini_extract_factory_review(doc_text)

        # Add Audit Date to the final data if found
        if audit_date:
            data["audit_date"] = audit_date

        if include_text_preview:
            return JSONResponse({
                "data": data,
                "preview": doc_text[:1000] + ("…[truncated]" if len(doc_text) > 1000 else ""),
                "meta": {
                    "pages": len(pages),
                    "used_ocr_pages": sum(1 for p in pages if p.get('used_ocr')),
                },
            })
        return JSONResponse(data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

