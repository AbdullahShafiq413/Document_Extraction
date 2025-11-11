# extractor.py — OCR + Gemini-only extraction with robust preprocessing for phone scans

import os, time, json
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()
try:
    # new recommended SDK
    
    from google import genai as genai_client
    NEW_GENAI = True
except Exception:
    # fallback: older "google.generativeai" package
    import google.generativeai as genai_client
    NEW_GENAI = False

import logging
logger = logging.getLogger("extractor")
logger.setLevel(logging.INFO)
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import pandas as pd
import google.generativeai as genai

# NEW: OpenCV for preprocessing (install: pip install opencv-python)
import cv2
import numpy as np

# ----------------------------
# Utilities
# ----------------------------
def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def _pix_to_pil(pix: fitz.Pixmap) -> Image.Image:
    if getattr(pix, "alpha", 0):
        pix = fitz.Pixmap(pix, 0)
    mode = "RGB" if pix.n >= 3 else "L"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return img

def _deskew_by_osd(pil_img: Image.Image) -> Image.Image:
    """Use Tesseract OSD to detect rotation; rotate to upright if needed."""
    try:
        osd = pytesseract.image_to_osd(pil_img)
        # osd contains lines like: "Rotate: 90\nOrientation in degrees: 90\n"
        angle = 0
        for line in osd.splitlines():
            if "Rotate:" in line:
                angle = int(line.split(":")[1].strip())
                break
        if angle and angle % 360 != 0:
            return pil_img.rotate(-angle, expand=True, fillcolor="white")
    except Exception:
        pass
    return pil_img

def _preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Heavy preprocessing for phone scans:
    - deskew (OSD)
    - grayscale + CLAHE (contrast)
    - de-shadow via background subtraction
    - adaptive threshold
    - mild denoise + unsharp
    """
    pil_img = _deskew_by_osd(pil_img)

    img = _pil_to_cv(pil_img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast boost (handles uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # De-shadow: estimate background via large blur, subtract
    bg = cv2.medianBlur(gray, 31)
    norm = cv2.divide(gray, bg, scale=255)

    # Adaptive threshold (robust to variable lighting)
    th = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )

    # Morphological open to clean small noise
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # Unsharp mask: sharpen text edges
    blur = cv2.GaussianBlur(opened, (0, 0), 1.0)
    sharp = cv2.addWeighted(opened, 1.5, blur, -0.5, 0)

    # Convert back to PIL (keep as 8-bit)
    return _cv_to_pil(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))

def _try_ocr_variants(pil_img: Image.Image, lang: str) -> Dict[str, Any]:
    """
    Run OCR multiple ways and choose the result with most characters.
    Variants:
      - original vs preprocessed
      - several PSMs: 6 (uniform text), 3 (auto), 4 (columns), 11 (sparse)
      - OEM auto
    """
    variants = []
    imgs = [("orig", pil_img), ("prep", _preprocess_for_ocr(pil_img))]

    psms = [6, 3, 4, 11]
    for tag, im in imgs:
        for psm in psms:
            try:
                cfg = f"--oem 3 --psm {psm}"
                text = pytesseract.image_to_string(im, lang=lang, config=cfg).strip()
                variants.append((len(text), f"{tag}_psm{psm}", text))
            except Exception:
                continue

    # fallback single run if all failed
    if not variants:
        text = pytesseract.image_to_string(pil_img, lang=lang, config="--psm 3").strip()
        variants.append((len(text), "fallback_psm3", text))

    # pick the longest text
    variants.sort(key=lambda x: x[0], reverse=True)
    best_len, best_tag, best_text = variants[0]
    return {"page_text": best_text, "debug": {"variant": best_tag, "length": best_len}}

def _ocr_page_image(pil_img: Image.Image, lang: str = "eng", psm: int = 3) -> Dict[str, Any]:
    """
    Backwards-compatible wrapper that now calls the stronger multi-variant OCR.
    (We keep 'psm' in the signature so your API remains compatible.)
    """
    best = _try_ocr_variants(pil_img, lang=lang)
    return {"page_text": best["page_text"], "lines": [], "words": []}

def _extract_native_text(page: fitz.Page) -> Dict[str, Any]:
    page_text = page.get_text("text").strip()
    return {"page_text": page_text, "blocks": []}

def extract_pdf_content(
    pdf_path: str,
    *,
    dpi: int = 300,
    lang: str = "eng",
    ocr_psm: int = 3,
    ocr_on_empty_only: bool = True,
    force_ocr: bool = False,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extract content per page. For image-heavy/phone scans, we auto-bump DPI and use robust OCR.
    """
    results: List[Dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        num_pages = len(doc)
        page_limit = min(num_pages, max_pages) if max_pages is not None else num_pages
        if page_limit <= 0:
            page_limit = num_pages

        for i in range(page_limit):
            page = doc[i]
            native = _extract_native_text(page)

            # Decide if OCR is needed
            need_ocr = force_ocr
            if not force_ocr and ocr_on_empty_only:
                if len(native["page_text"]) < 10:
                    need_ocr = True
            elif not ocr_on_empty_only:
                need_ocr = True

            ocr = None
            if need_ocr:
                # Phone scans benefit from higher DPI: bump to at least 400
                target_dpi = max(dpi, 400)
                zoom = target_dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                pil_img = _pix_to_pil(pix)

                # Robust OCR
                ocr = _ocr_page_image(pil_img, lang=lang, psm=ocr_psm)

            # Choose best source
            native_len = len(native["page_text"])
            ocr_len = len(ocr["page_text"]) if ocr else 0

            if force_ocr:
                final_text = ocr["page_text"] if ocr else native["page_text"]; used_ocr = bool(ocr)
            elif ocr_on_empty_only:
                if ocr and ocr_len > 0:
                    final_text = ocr["page_text"]; used_ocr = True
                else:
                    final_text = native["page_text"]; used_ocr = False
            else:
                if ocr_len > native_len:
                    final_text = ocr["page_text"]; used_ocr = True
                else:
                    final_text = native["page_text"]; used_ocr = False

            results.append({
                "page_index": i,
                "final_text": (final_text or "").strip(),
                "used_ocr": used_ocr,
            })
    return results

def build_single_text(pages: List[Dict[str, Any]]) -> str:
    return "\n".join(p["final_text"] for p in pages if p["final_text"].strip())

# ----------------------------
# Gemini-only extraction (unchanged)
# ----------------------------
GEMINI_MODELS_ORDER = ["gemini-2.5-pro", "gemini-2.5-flash"]

SCHEMA_PROMPT = """
You are an information extraction system. Your job is to read audit reports
(SMETA, BSCI, ETI, etc.) and RETURN ONLY A SINGLE JSON OBJECT with specific
fields for a Wage Tracker module.

CRITICAL RULES

- Output MUST be a single, valid JSON object. No markdown, no explanations.
- Do NOT include any keys other than the ones listed below.
- If information is missing or not clearly stated, use null for that field.
- Do NOT guess or invent numbers or text.
- Preserve currency symbols and units exactly as in the document when possible,
  e.g. "RMB 3,902", "85 hours/month".
- It is OK if values are text with extra explanation, e.g.
  "RMB 2,251 (actual per month wage for standard hours)" or
  "7 total management and office staff".

The JSON object MUST have exactly these 12 keys, with values of type
string or null:

{
  "data_year_month": string | null,
  "lowest_monthly_wage_gross": string | null,
  "average_monthly_wage_gross": string | null,
  "average_women_monthly_wage_gross": string | null,
  "average_men_monthly_wage_gross": string | null,
  "average_contracted_hours": string | null,
  "average_extra_hours_per_month": string | null,
  "amount_women_workers": string | null,
  "amount_men_workers": string | null,
  "amount_women_managers": string | null,
  "amount_men_managers": string | null,
  "amount_migrant_managers": string | null
}

TARGET STYLE EXAMPLE (do NOT output this in your answer; it is only a guide):

- data_year_month: "2023-08-31"
- lowest_monthly_wage_gross: "RMB 3,902"
- average_monthly_wage_gross: "RMB 2,251 (actual per month wage for standard hours)"
- average_women_monthly_wage_gross: null
- average_men_monthly_wage_gross: null
- average_contracted_hours: "8 hours/day, 40 hours/week, 184 hours/month"
- average_extra_hours_per_month: "85 hours/month"
- amount_women_workers: "53"
- amount_men_workers: "41"
- amount_women_managers: null
- amount_men_managers: "7 total management and office staff"
- amount_migrant_managers: null

FIELD-BY-FIELD INSTRUCTIONS

1) data_year_month
- Use the main audit date / report reference date that represents the data period,
  usually shown as "Report reference / Start Date / End Date".
- Prefer the single date associated with the current audit period (not the non-compliance dates).
- Format as: YYYY-MM-DD (e.g. "2023-08-31").
- If you cannot clearly identify the correct date, set to null.

2) lowest_monthly_wage_gross
- In SMETA: look in the "Wages Analysis" or "Summary information" section.
- Prefer the text of "Lowest actual wages found: ... per month" or equivalent phrase.
- If that phrase is not present, use the lowest actual wage value for standard/contracted
  hours per month if clearly indicated.
- Preserve currency and wording exactly, e.g. "RMB 3,902" or "RMB4000 per month".
- If multiple currencies/months are shown, choose the value that matches the main,
  current audit period/month; otherwise set to null.

3) average_monthly_wage_gross
- In SMETA: use the "Actual Per Month" value under
  "Wage for standard/contracted hours" for wages.
- Convert to a string with currency, e.g. "RMB 2,251".
- If helpful, you may add a short explanation in parentheses, for example:
  "RMB 2,251 (actual per month wage for standard hours)".
- If the table only gives a numeric value (e.g. 2251.0) but it is clearly
  RMB per month, you may format it as "RMB 2,251".
- If no clear average monthly wage is given, use null.

4) average_women_monthly_wage_gross
- Only fill this if the report clearly provides average or typical monthly wages
  for women separately (e.g. gender-segmented wage table).
- Return the value as text with currency and units (e.g. "RMB 3,000 per month").
- If there is no explicit women-only average wage, set this field to null.

5) average_men_monthly_wage_gross
- Only fill this if the report clearly provides average or typical monthly wages
  for men separately.
- Return the value as text with currency and units.
- If there is no explicit men-only average wage, set this field to null.

6) average_contracted_hours
- Use the standard/contracted working hours from the summary table.
  In SMETA this is usually shown under:
  "Standard/Contracted work hours (Maximum legal and actual required working
   hours excluding overtime, per day, week, and month)".
- Use the ACTUAL values (not the legal maximum) if available.
- Combine day/week/month in one string if possible, e.g.:
  "8 hours/day, 40 hours/week, 184 hours/month".
- If only some units are available, include only those (e.g. "8 hours/day, 40 hours/week").
- If no clear contracted hours are given, set to null.

7) average_extra_hours_per_month
- Use the overtime "Actual Per Month" value from the "Overtime hours" section.
- Express it as a string with units:
  - Example: "85 hours/month" if the actual overtime per month is 85.0 hours.
- If there are multiple sample months, choose the value that matches the main,
  current audit month, or the typical figure if the report clearly identifies it.
- If overtime per month cannot be clearly determined, set this field to null.

8) amount_women_workers
- Use the total number of female workers in the facility.
- In SMETA "Worker Analysis" tables, sum or read directly from the "Total" column
  for female workers (local + migrant + home workers).
- Example: if the table shows 53 total female workers, return "53".
- Do NOT use the number of interviewed workers; use the total number of workers.
- If you cannot clearly identify the total number of women workers, set this to null.

9) amount_men_workers
- Use the total number of male workers in the facility.
- In SMETA "Worker Analysis" tables, read from the "Total" column for male workers.
- Example: if the table shows 41 total male workers, return "41".
- Do NOT use the number of interviewed workers.
- If you cannot clearly identify the total number of men workers, set this to null.

10) amount_women_managers
- Look for any section that breaks down management by gender, for example in BSCI:
  "Management - Female: 178 workers".
- If such a breakdown exists, return that number as a string (e.g. "178").
- If the report only gives a total number of managers without specifying female
  managers, set this field to null.
- Do NOT guess gender distribution.

11) amount_men_managers
- Prefer explicit gender breakdowns, e.g. "Management - Male: 179 workers" in BSCI.
  In that case return "179".
- In reports where the only information is a combined phrase like
  "7 management and office staff" and there is NO gender breakdown:
  - You MAY return the exact phrase for this field, e.g. "7 total management and office staff",
    and set amount_women_managers and amount_migrant_managers to null.
- Never invent a men/women split when it is not given.

12) amount_migrant_managers
- Use this only if the report explicitly provides the number of migrant
  managers or identifies managers by migrant status.
- If there is no explicit count of migrant managers, set this to null.
- Do NOT derive migrant managers from general migrant worker numbers.

MISSING OR AMBIGUOUS DATA

- If a value cannot be confidently determined from the text, use null.
- If multiple numbers appear and it is not clear which one should be used for
  the field, use null rather than guessing.

FINAL OUTPUT FORMAT

Return ONLY a single JSON object like this (structure only, values filled in):

{
  "data_year_month": "... or null",
  "lowest_monthly_wage_gross": "... or null",
  "average_monthly_wage_gross": "... or null",
  "average_women_monthly_wage_gross": "... or null",
  "average_men_monthly_wage_gross": "... or null",
  "average_contracted_hours": "... or null",
  "average_extra_hours_per_month": "... or null",
  "amount_women_workers": "... or null",
  "amount_men_workers": "... or null",
  "amount_women_managers": "... or null",
  "amount_men_managers": "... or null",
  "amount_migrant_managers": "... or null"
}
"""


def _gemini_call(prompt: str, model_name: str, retries: int = 4, timeout: int = 60) -> Optional[str]:
    """
    New-style call to Google GenAI. Uses google-genai if available; falls back to older package.
    Returns response.text on success, else None.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in environment.")

    last_err = None

    for attempt in range(retries):
        try:
            if NEW_GENAI:
                # new client usage
                # pip install google-genai
                client = genai_client.Client(api_key=api_key)
                # 'contents' accepts string or list (multimodal)
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    # generation parameters can be added here if needed:
                    # max_output_tokens=800,
                )
                text = getattr(resp, "text", None)
                if text is None:
                    # some responses have a different structure; try to stringify
                    logger.info("Raw response object: %s", resp)
                    text = str(resp)
                logger.info("Gemini (%s) success (attempt %d)", model_name, attempt+1)
                return text

            else:
                # fallback to old API (your previous style)
                genai_client.configure(api_key=api_key)
                model = genai_client.GenerativeModel(model_name)
                resp = model.generate_content(prompt)
                text = getattr(resp, "text", None)
                if text is None:
                    logger.info("Old SDK raw response: %s", resp)
                    text = str(resp)
                logger.info("Gemini (old sdk) success (attempt %d)", attempt+1)
                return text

        except Exception as e:
            last_err = e
            logger.warning("Gemini call failed (model=%s attempt=%d): %s", model_name, attempt+1, e)
            # exponential backoff
            import time
            time.sleep(min(2 ** attempt, 16))

    logger.error("Gemini call failed after %d attempts. Last error: %s", retries, last_err)
    return None

def _clean_json_text(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return s

def _chunk_text(text: str, max_chars: int) -> List[str]:
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def gemini_extract_fields_only(doc_text: str, max_chars: int = 120_000) -> Dict[str, Any]:
    if len(doc_text) <= max_chars:
        prompt = SCHEMA_PROMPT + "\n\nDOCUMENT TEXT:\n" + doc_text
        for m in GEMINI_MODELS_ORDER:
            out = _gemini_call(prompt, m)
            if out:
                out = _clean_json_text(out)
                try:
                    return json.loads(out)
                except Exception:
                    repaired = _gemini_call(
                        f"Return ONLY strict JSON valid for the required schema. Fix to valid JSON:\n{out}", m
                    )
                    if repaired:
                        repaired = _clean_json_text(repaired)
                        return json.loads(repaired)
        raise RuntimeError("Gemini calls failed or returned invalid JSON.")
    # chunked
    chunks = _chunk_text(doc_text, max_chars)
    partials = []
    for idx, ch in enumerate(chunks):
        p = SCHEMA_PROMPT + f"\n\nDOCUMENT TEXT (PART {idx+1}/{len(chunks)}):\n" + ch
        p += "\n\nIf a field is not present in this PART, return null for that field."
        got = None
        for m in GEMINI_MODELS_ORDER:
            got = _gemini_call(p, m)
            if got:
                got = _clean_json_text(got)
                try:
                    partials.append(json.loads(got))
                    break
                except Exception:
                    repaired = _gemini_call(f"Return STRICT JSON only. Repair to valid JSON:\n{got}", m)
                    if repaired:
                        repaired = _clean_json_text(repaired)
                        partials.append(json.loads(repaired))
                        break
        if not got:
            raise RuntimeError("Gemini failed during chunked map step.")
    combine_prompt = f"""{SCHEMA_PROMPT}

Below are JSON candidates extracted from different chunks of the same document.
Merge them into ONE final STRICT JSON answer:
- Prefer explicit, consistent values.
- Deduplicate arrays by exact string match.
- If no candidate provides a value, return null.

CANDIDATE JSONS:
{json.dumps(partials, ensure_ascii=False, indent=2)}
"""
    for m in GEMINI_MODELS_ORDER:
        combined = _gemini_call(combine_prompt, m)
        if combined:
            combined = _clean_json_text(combined)
            try:
                return json.loads(combined)
            except Exception:
                repaired = _gemini_call(f"Return STRICT JSON only. Repair to valid JSON:\n{combined}", m)
                if repaired:
                    repaired = _clean_json_text(repaired)
                    return json.loads(repaired)
    raise RuntimeError("Gemini failed to combine chunked results into valid JSON.")
