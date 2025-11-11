# Wage Tracker PDF Extractor (Gemini + FastAPI)

A tiny FastAPI service that ingests **social/audit PDFs** (SMETA, BSCI, ETI-style reports), extracts text using native PDF text or OCR (for scans/phone photos), and then asks **Google Gemini** to return a **strict JSON** payload for a Wage Tracker module.

This version is specialised for extracting:

- Data Year/Month  
- Wages (lowest and average)  
- Working hours (contracted + overtime)  
- Worker and manager counts by gender  
- Migrant managers (if explicitly given)

Core logic lives in `extractor.py`; the HTTP API is in `main.py`. Dependencies are listed in `requirements.txt`.

---

## What it does (at a glance)

- Reads PDFs page-by-page with **PyMuPDF**.
- If a page has little or no extractable text, it switches to **Tesseract OCR** with heavy **OpenCV** preprocessing:
  - deskew (Tesseract OSD)
  - de-shadow via background subtraction
  - CLAHE contrast boost
  - adaptive threshold
  - denoise + sharpen
- Stitches page text together and sends it to **Gemini 2.5** (`gemini-2.5-pro` → `gemini-2.5-flash` fallback) with a **strict schema prompt** for Wage Tracker fields.
- Exposes a minimal HTTP API with **`/health`** and **`/extract`** endpoints.

---

## Project layout

```text
.
├── main.py          # FastAPI app & HTTP endpoints
├── extractor.py     # OCR + preprocessing + Gemini-based field extraction
└── requirements.txt # Python dependencies
````

---

## Extracted fields (Wage Tracker schema)

The API returns a single JSON object with **exactly these keys**.
All values are **string or null** so that currency symbols, units and explanatory text are preserved.

```json
{
  "data_year_month": "2023-08-31",
  "lowest_monthly_wage_gross": "RMB 3,902",
  "average_monthly_wage_gross": "RMB 2,251 (actual per month wage for standard hours)",
  "average_women_monthly_wage_gross": null,
  "average_men_monthly_wage_gross": null,
  "average_contracted_hours": "8 hours/day, 40 hours/week, 184 hours/month",
  "average_extra_hours_per_month": "85 hours/month",
  "amount_women_workers": "53",
  "amount_men_workers": "41",
  "amount_women_managers": null,
  "amount_men_managers": "7 total management and office staff",
  "amount_migrant_managers": null
}
```



---

## Requirements

* **Python 3.10+** (recommended)
* **Tesseract OCR** installed on the system and available on `PATH`

  * Windows: install from the UB Mannheim build
  * Linux: `sudo apt install tesseract-ocr`
* A **Google API key** for Gemini models (Gemini 2.5)
* Poppler is **not** required (we use PyMuPDF, not pdf2image)

Python packages are listed in `requirements.txt`.

---

## Environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

The `/health` endpoint will tell you whether the key is detected:

```json
{ "ok": true, "gemini_key_present": true }
```

---

## Setup & run

From the project folder:

```bash
# 1) Create & activate a virtualenv
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
# source venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Create .env with your Google API key
echo GOOGLE_API_KEY=your_gemini_api_key_here > .env   # (or create manually on Windows)

# 4) Run the FastAPI app
uvicorn main:app --reload --port 8000
```

The service will start at:

* `http://127.0.0.1:8000` (root – no route defined, so 404 is normal)
* `http://127.0.0.1:8000/health` (health check)
* `http://127.0.0.1:8000/docs` (interactive Swagger UI)

---

## API

### `GET /health`

Simple health check.

**Response:**

```json
{
  "ok": true,
  "gemini_key_present": true
}
```

---

### `POST /extract` (multipart/form-data)

Upload a PDF and configure extraction options.

**Form fields** (all except `file` are optional):

* `file` (**required**): the PDF to process (must end with `.pdf`)
* `lang` (default: `"eng"`): Tesseract language code(s) for OCR
* `dpi` (default: `300`): base render DPI (automatically bumped to ≥ 400 for OCR pages)
* `ocr_psm` (default: `3`): Tesseract page segmentation mode (internally we try multiple variants)
* `force_ocr` (default: `false`): force OCR on every page even if native text exists
* `ocr_on_empty_only` (default: `true`): only OCR pages with little/no native text
* `max_pages` (default: empty/None): limit number of processed pages; empty = all
* `include_text_preview` (default: `false`): if true, response includes a short text preview + meta

**Example (via Swagger `/docs`):**

1. Open `http://127.0.0.1:8000/docs`
2. Expand `POST /extract`
3. Click **Try it out**
4. Upload a PDF SMETA/BSCI report as `file`
5. In max_pages section make it Empty.
6. Click **Execute**

**Example JSON response:**

```json
{
  "data_year_month": "2023-08-31",
  "lowest_monthly_wage_gross": "RMB 3,902",
  "average_monthly_wage_gross": "RMB 2,251 (actual per month wage for standard hours)",
  "average_women_monthly_wage_gross": null,
  "average_men_monthly_wage_gross": null,
  "average_contracted_hours": "8 hours/day, 40 hours/week, 184 hours/month",
  "average_extra_hours_per_month": "85 hours/month",
  "amount_women_workers": "53",
  "amount_men_workers": "41",
  "amount_women_managers": null,
  "amount_men_managers": "7 total management and office staff",
  "amount_migrant_managers": null
}
```

If `include_text_preview = true`, the response is wrapped like:

```json
{
  "data": { ...fields above... },
  "preview": "First 1000 chars of extracted text...",
  "meta": {
    "pages": 6,
    "used_ocr_pages": 3
  }
}
```

