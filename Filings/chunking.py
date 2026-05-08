"""
Chunking and cleaning the JSON files . 

"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    from capstone_rag_project.Filings.config_loader import load_config_yaml
except ModuleNotFoundError:
    from config_loader import load_config_yaml


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
logger = logging.getLogger("chunking")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_config_yaml() -> Dict[str, Any]:
    data = load_config_yaml(CONFIG_PATH)
    if not data:
        logger.info("config.yaml not found/empty at %s. Using defaults.", CONFIG_PATH)
    return data


_CONFIG = _load_config_yaml()


def _cfg_int(key: str, default: int) -> int:
    raw = str(_CONFIG.get(key, default)).strip()
    if "=" in raw:
        raw = raw.split("=", 1)[1].strip()
    return int(raw.strip('"').strip("'"))


MIN_CHUNK_WORDS = _cfg_int("MIN_CHUNK_WORDS", 150)
MAX_CHUNK_WORDS = _cfg_int("MAX_CHUNK_WORDS", 450)
CHUNK_OVERLAP_WORDS = _cfg_int("CHUNK_OVERLAP_WORDS", 80)
BAD_EXACT_HEADINGS = {
    "Number of",
    "Average Price Paid Per Share",
    "Weighted-Average Grant Date Fair Value Per RSU",
    "Notional Amount",
    "Total Number of Shares Purchased",
}
MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
BAD_FINAL_EXACT_HEADINGS = {
    "Total",
    "Total Fair Value",
    "Credit Risk Amount",
    "Fair Value",
    "Fixed-rate debt",
    "Services",
    "Filing Date/ Period End Date",
    "Fiscal Period",
    "Under the Plans or Programs",
    "Weighted-Average Grant-Date Fair Value Per RSU",
    "Total Number of Shares Purchased as Part of Publicly Announced Plans or Programs",
}


# -----------------------------
# HELPERS
# -----------------------------

def is_item_heading(text: str) -> bool:
    text = normalize_whitespace(text)
    return bool(re.match(r"^Item\s+\d+(?:\.\d+)?[A-Z]?\.?\s*.*$", text, flags=re.IGNORECASE))

def normalize_whitespace(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(text.split())

def looks_like_date_heading(text: str) -> bool:
    text = normalize_whitespace(text)

    patterns = [
        r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}$",
        r"^For the quarterly period ended\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}$",
        r"^For the fiscal year ended\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}$",
        r"^\d{4}$",
    ]
    return any(re.match(p, text, flags=re.IGNORECASE) for p in patterns)


def looks_like_table_label(text: str) -> bool:
    text = normalize_whitespace(text)

    patterns = [
        r"^Total(\s*\(\d+\))*$",
        r"^Total Fair Value$",
        r"^Fair Value$",
        r"^Credit Risk Amount$",
        r"^Notional Amount$",
        r"^Adjusted Cost$",
        r"^Unrealized Gains$",
        r"^Unrealized Losses$",
        r"^Fixed-rate debt(\s*\(\d+\))*$",
        r"^Services(\s*\(\d+\))*$",
        r"^Derivative assets(\s*\(\d+\))*:?\s*$",
        r"^Derivative liabilities(\s*\(\d+\))*:?\s*$",
        r"^Cash and Cash Equivalents$",
        r"^Current Marketable Securities$",
        r"^Non-Current Marketable Securities$",
        r"^Effective Interest Rate$",
        r"^Weighted-Average Grant-Date Fair Value Per RSU$",
        r"^Filing Date/?\s*Period End Date$",
        r"^Fiscal Period$",
        r"^Under the Plans or Programs(\s*\(\d+\))*$",
        r"^Total Number of Shares Purchased as Part of Publicly Announced Plans or Programs$",
    ]
    return any(re.match(p, text, flags=re.IGNORECASE) for p in patterns)


def looks_like_bad_subsection_heading(text: str) -> bool:
    text = normalize_whitespace(text)

    if not text:
        return True

    if looks_like_date_heading(text):
        return True

    if looks_like_table_label(text):
        return True

    if re.fullmatch(r"\d+(\.\d+)?", text):
        return True

    if word_count(text) == 1 and len(text) < 8:
        return True

    return False


def is_meaningful_subsection_heading(text: str) -> bool:
    text = normalize_whitespace(text)

    if looks_like_bad_subsection_heading(text):
        return False

    if text.endswith("."):
        return False

    if 1 <= word_count(text) <= 12 and len(text) <= 120:
        return True

    return False


def repair_heading_ocr_spacing(text: str) -> str:
    """
    Repair common OCR/spacing artifacts in short heading-like strings.
    Examples:
      - "I nsider Trading Policy" -> "Insider Trading Policy"
      - "Reco very Pol icy" -> "Recovery Policy"
    """
    text = normalize_whitespace(text)
    if not text:
        return text

    # Case 1: single-letter prefix split from word.
    repaired = re.sub(r"\b([A-Za-z])\s+([a-z]{2,})\b", r"\1\2", text)

    # Case 2: broken word split into two short tokens (e.g., "Reco very", "Pol icy").
    tokens = repaired.split()
    if len(tokens) < 2:
        return repaired

    pair_indices: List[int] = []
    for i in range(len(tokens) - 1):
        left = tokens[i]
        right = tokens[i + 1]
        if re.match(r"^[A-Z][a-z]{1,4}$", left) and re.match(r"^[a-z]{2,4}$", right):
            pair_indices.append(i)

    # Be conservative: merge aggressive pairs only when pattern is repeated.
    merge_set = set()
    if len(pair_indices) >= 2:
        merge_set.update(pair_indices)
    elif len(pair_indices) == 1:
        i = pair_indices[0]
        if len(tokens[i]) <= 3 or len(tokens[i + 1]) <= 3:
            merge_set.add(i)

    if not merge_set:
        return repaired

    merged_tokens: List[str] = []
    i = 0
    while i < len(tokens):
        if i in merge_set and i + 1 < len(tokens):
            merged_tokens.append(tokens[i] + tokens[i + 1])
            i += 2
            continue
        merged_tokens.append(tokens[i])
        i += 1

    return " ".join(merged_tokens)

def looks_like_page_footer(text: str, company_title: Optional[str] = None, form_type: Optional[str] = None) -> bool:
    text = normalize_whitespace(text)
    escaped_company = re.escape(company_title.strip()) if company_title and company_title.strip() else None
    escaped_form = re.escape(form_type.strip().upper()) if form_type and form_type.strip() else None

    patterns = [
        # Generic issuer footer pattern: "<Issuer> | ... Form 10-K/Q ... | <page>"
        r"^[A-Za-z0-9 .,&'()/-]+\s*\|\s*.*Form\s+10-[QK]\s*\|\s*\d+$",
        # Generic quarter/year variant often seen in filings.
        r"^[A-Za-z0-9 .,&'()/-]+\s*\|\s*(Q[1-4]|FY)\s*\d{4}\s+Form\s+10-[QK]\s*\|\s*\d+$",
    ]
    if escaped_company and escaped_form:
        patterns.append(
            rf"^{escaped_company}\s*\|\s*.*Form\s+{escaped_form}\s*\|\s*\d+$"
        )
    elif escaped_company:
        patterns.append(
            rf"^{escaped_company}\s*\|\s*.*Form\s+10-[QK]\s*\|\s*\d+$"
        )

    return any(re.match(p, text, flags=re.IGNORECASE) for p in patterns)


def looks_like_image_caption(text: str) -> bool:
    return bool(re.search(r"\.(jpg|jpeg|png|gif|bmp|webp)$", text, flags=re.IGNORECASE))


def looks_like_checkbox_line(text: str) -> bool:
    return "â˜’" in text or "â˜" in text


def looks_like_toc_page_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}", text.strip()))


def looks_like_filing_boilerplate(text: str) -> bool:
    boilerplate_patterns = [
        r"^UNITED STATES$",
        r"^SECURITIES AND EXCHANGE COMMISSION$",
        r"^Washington,\s*D\.C\.\s*\d{5}$",
        r"^FORM 10-[QK]$",
        r"^Commission File Number:",
        r"^Indicate by check mark",
        r"^Yes\s*[â˜’â˜]\s*No\s*[â˜’â˜]$",
        r"^For the transition period from to \.$",
        r"^\(Mark One\)$",
        r"^\(Registrant's telephone number, including area code\)$",
        r"^\(Exact name of Registrant as specified in its charter\)$",
        r"^TABLE OF CONTENTS$",
        r"^Part I$",
        r"^Part II$",
        r"^FORM 8-K$",
        r"^CURRENT REPORT$",
        r"^Pursuant to Section 13 OR 15\(d\) of The Securities Exchange Act of 1934$",
        r"^Date of Report",
        r"^Commission File Number",
        r"^State or other jurisdiction of incorporation",
        r"^I\.R\.S\. Employer Identification No\.",
        r"^Title of each class",
        r"^Trading Symbol",
        r"^Name of each exchange on which registered",
        r"^Check the appropriate box below",
        r"^Written communications pursuant to Rule 425",
        r"^Soliciting material pursuant to Rule 14a-12",
        r"^Pre-commencement communications pursuant to Rule 14d-2",
        r"^Pre-commencement communications pursuant to Rule 13e-4",
    ]
    return any(re.match(p, text, flags=re.IGNORECASE) for p in boilerplate_patterns)


def is_probably_noise(
    text: str,
    label: str,
    content_layer: str,
    company_title: Optional[str] = None,
    form_type: Optional[str] = None,
) -> bool:
    text = normalize_whitespace(text)

    if not text:
        return True

    if content_layer == "furniture":
        return True

    if label in {"caption"} and looks_like_image_caption(text):
        return True

    if looks_like_page_footer(text, company_title=company_title, form_type=form_type):
        return True

    if looks_like_checkbox_line(text):
        return True

    if looks_like_toc_page_number(text):
        return True

    if looks_like_filing_boilerplate(text):
        return True

    # Skip tiny fragments that carry almost no semantic value
    if word_count(text) <= 1 and not (is_item_heading(text) or re.match(r"^Note\s+\d+", text, flags=re.IGNORECASE)):
        return True

    return False


def is_section_heading(text: str) -> bool:
    text = normalize_whitespace(text)
    if text in BAD_EXACT_HEADINGS:
        return False
    patterns = [
        r"^PART\s+[IVXLC]+\b.*",
        r"^Item\s+\d+(?:\.\d+)?[A-Z]?\.?\s*.*$",
        r"^Note\s+\d+\s*[-â€“]\s+.+",
        r"^[A-Z][A-Za-z0-9,'&/() \-]{3,80}$",
    ]
    if any(re.match(p, text, flags=re.IGNORECASE) for p in patterns):
        return True
    if is_item_heading(text):
        return True

    # All-caps heading heuristic
    alpha = re.sub(r"[^A-Za-z]", "", text)
    if alpha and len(alpha) > 6 and text.upper() == text and len(text) < 120:
        return True

    return False


def classify_heading(text: str) -> str:
    text = normalize_whitespace(text)

    if is_item_heading(text):
        return "item"
    if re.match(r"^PART\s+[IVXLC]+\b", text, flags=re.IGNORECASE):
        return "part"
    if re.match(r"^Item\s+\d+(?:\.\d+)?[A-Z]?\.?\s*.*$", text, flags=re.IGNORECASE):
        return "item"
    if re.match(r"^Note\s+\d+\s*[-â€“]", text, flags=re.IGNORECASE):
        return "note"
    return "subsection"


def normalize_heading_text(text: str) -> str:
    text = re.sub(r"\s*\(\d+\)(\(\d+\))*\s*$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_subsection_title(text: str) -> str | None:
    if not text:
        return None

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*\(\d+\)(\(\d+\))*\s*$", "", text).strip()
    text = text.replace("â€“", "-")

    if is_bad_final_subsection_title(text):
        return None

    return text


def normalize_subsection_title(text: str) -> str | None:
    if not text:
        return None
    text = normalize_whitespace(text)
    normalization_map = {
        "Foreign Exchange Risk": "Foreign Exchange Rate Risk",
        "Cash, cash equivalents and marketable securities":
            "Cash, Cash Equivalents and Marketable Securities",
    }
    text = normalization_map.get(text, text)
    return text


def get_valid_subsection_title(text: str) -> str | None:
    cleaned_subsection = clean_subsection_title(text)
    normalized_subsection = normalize_subsection_title(cleaned_subsection)
    if normalized_subsection and not is_bad_final_subsection_title(normalized_subsection):
        return normalized_subsection
    return None


def is_bad_final_subsection_title(text: str) -> bool:
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return True

    if text in BAD_FINAL_EXACT_HEADINGS:
        return True

    if re.match(
        rf"^For the quarterly period ended {MONTHS}\s+\d{{1,2}},\s+\d{{4}}$",
        text,
        re.IGNORECASE,
    ):
        return True

    if re.match(
        rf"^For the fiscal year ended {MONTHS}\s+\d{{1,2}},\s+\d{{4}}$",
        text,
        re.IGNORECASE,
    ):
        return True

    if re.match(rf"^{MONTHS}\s+\d{{1,2}},\s+\d{{4}}$", text, re.IGNORECASE):
        return True

    if re.match(r"^\d{4}$", text):
        return True

    return False


def chunk_words(text: str, max_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def build_chunk_text(
    part_title: Optional[str],
    item_title: Optional[str],
    note_title: Optional[str],
    subsection_title: Optional[str],
    content: str
) -> str:
    lines = []
    if part_title:
        lines.append(f"Part: {part_title}")
    if item_title:
        lines.append(f"Item: {item_title}")
    if note_title:
        lines.append(f"Note: {note_title}")
    if subsection_title:
        lines.append(f"Subsection: {subsection_title}")
    lines.append("")
    lines.append(content.strip())
    return "\n".join(lines).strip()


# -----------------------------
# DOCLING EXTRACTION
# -----------------------------
def extract_text_nodes(docling_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    texts = docling_json.get("docling", {}).get("texts", [])
    extracted = []

    for idx, node in enumerate(texts):
        text = normalize_whitespace(node.get("text", ""))
        label = node.get("label", "")
        content_layer = node.get("content_layer", "")
        extracted.append({
            "order": idx,
            "text": text,
            "label": label,
            "content_layer": content_layer
        })

    logger.info("Extracted text nodes: %s", len(extracted))
    return extracted


def clean_text_nodes(
    nodes: List[Dict[str, Any]],
    company_title: Optional[str] = None,
    form_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    cleaned = []
    for node in nodes:
        text = node["text"]
        label = node["label"]
        layer = node["content_layer"]

        if is_probably_noise(
            text,
            label,
            layer,
            company_title=company_title,
            form_type=form_type,
        ):
            continue

        cleaned.append(node)

    logger.info("Cleaned text nodes: %s (dropped=%s)", len(cleaned), len(nodes) - len(cleaned))
    return cleaned


# -----------------------------
# SEMANTIC BLOCK BUILDING
# -----------------------------
def build_semantic_blocks(
    nodes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    blocks = []

    current_part = None
    current_item = None
    current_note = None
    current_subsection_raw = None
    current_subsection_clean = None
    current_subsection_normalized = None

    current_block_lines: List[str] = []
    current_block_meta: Dict[str, Optional[str]] = {}

    def flush_block():
        nonlocal current_block_lines, current_block_meta, blocks
        content = normalize_whitespace(" ".join(current_block_lines))
        if content and word_count(content) >= 20:
            blocks.append({
                "part_title": current_block_meta.get("part_title"),
                "item_title": current_block_meta.get("item_title"),
                "note_title": current_block_meta.get("note_title"),
                "subsection_title_raw": current_block_meta.get("subsection_title_raw"),
                "subsection_title_clean": current_block_meta.get("subsection_title_clean"),
                "subsection_title_normalized": current_block_meta.get("subsection_title_normalized"),
                "content": content
            })
        current_block_lines = []

    for node in nodes:
        text = node["text"]
        heading_text = repair_heading_ocr_spacing(text)

        if is_section_heading(heading_text):
            heading_type = classify_heading(heading_text)

            flush_block()

            if heading_type == "part":
                current_part = normalize_heading_text(heading_text)
                current_item = None
                current_note = None
                current_subsection_raw = None
                current_subsection_clean = None
                current_subsection_normalized = None
            elif heading_type == "item":
                current_item = normalize_heading_text(heading_text)
                current_note = None
                current_subsection_raw = None
                current_subsection_clean = None
                current_subsection_normalized = None
            elif heading_type == "note":
                current_note = normalize_heading_text(heading_text)
                current_subsection_raw = None
                current_subsection_clean = None
                current_subsection_normalized = None
            elif heading_type == "subsection":
                raw_subsection = normalize_heading_text(heading_text)
                clean_subsection = clean_subsection_title(raw_subsection)
                normalized_subsection = normalize_subsection_title(clean_subsection)
                if normalized_subsection and not is_bad_final_subsection_title(normalized_subsection):
                    current_subsection_raw = raw_subsection
                    current_subsection_clean = clean_subsection
                    current_subsection_normalized = normalized_subsection
                else:
                    current_block_lines.append(text)
                    continue
            else:
                raw_subsection = normalize_heading_text(heading_text)
                clean_subsection = clean_subsection_title(raw_subsection)
                current_subsection_raw = raw_subsection
                current_subsection_clean = clean_subsection
                current_subsection_normalized = normalize_subsection_title(clean_subsection)

            current_block_meta = {
                "part_title": current_part,
                "item_title": current_item,
                "note_title": current_note,
                "subsection_title_raw": current_subsection_raw,
                "subsection_title_clean": current_subsection_clean,
                "subsection_title_normalized": current_subsection_normalized,
            }
            continue

        if not current_block_meta:
            current_block_meta = {
                "part_title": current_part,
                "item_title": current_item,
                "note_title": current_note,
                "subsection_title_raw": current_subsection_raw,
                "subsection_title_clean": current_subsection_clean,
                "subsection_title_normalized": current_subsection_normalized,
            }

        current_block_lines.append(text)

    flush_block()
    logger.info("Built semantic blocks: %s", len(blocks))
    return blocks


# -----------------------------
# TABLE SUMMARY PLACEHOLDER
# -----------------------------
def detect_table_intro(text: str) -> bool:
    patterns = [
        r"^The following table",
        r"^The following tables",
        r"^Net sales disaggregated",
        r"^A reconciliation of",
        r"^The following table shows",
    ]
    return any(re.match(p, text, flags=re.IGNORECASE) for p in patterns)


def _table_cells_to_matrix(table_data: Dict[str, Any]) -> List[List[str]]:
    num_rows = int(table_data.get("num_rows", 0) or 0)
    num_cols = int(table_data.get("num_cols", 0) or 0)
    if num_rows <= 0 or num_cols <= 0:
        return []

    matrix: List[List[str]] = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in table_data.get("table_cells", []):
        if not isinstance(cell, dict):
            continue
        r = int(cell.get("start_row_offset_idx", -1))
        c = int(cell.get("start_col_offset_idx", -1))
        if 0 <= r < num_rows and 0 <= c < num_cols:
            matrix[r][c] = normalize_whitespace(cell.get("text", ""))
    return matrix


def _choose_table_header_row(matrix: List[List[str]]) -> int:
    for i, row in enumerate(matrix[:3]):
        non_empty = [x for x in row if normalize_whitespace(x)]
        if len(non_empty) >= max(1, len(row) // 2):
            return i
    return 0


def _table_to_structured_payload(table_obj: Dict[str, Any]) -> Dict[str, Any]:
    data = table_obj.get("data", {}) if isinstance(table_obj, dict) else {}
    matrix = _table_cells_to_matrix(data if isinstance(data, dict) else {})
    if not matrix:
        return {"headers": [], "rows": [], "num_rows": 0, "num_cols": 0}

    header_row_idx = _choose_table_header_row(matrix)
    headers = [normalize_whitespace(h) or f"col_{i+1}" for i, h in enumerate(matrix[header_row_idx])]

    rows: List[Dict[str, str]] = []
    for row in matrix[header_row_idx + 1:]:
        if not any(normalize_whitespace(x) for x in row):
            continue
        rows.append({headers[i]: normalize_whitespace(row[i]) for i in range(min(len(headers), len(row)))})

    return {
        "headers": headers,
        "rows": rows,
        "num_rows": len(matrix),
        "num_cols": len(headers),
    }


def extract_structured_table_chunks(
    doc: Dict[str, Any],
    source_doc: str,
    filename: str,
    ticker: str,
    year: int,
    form_type: str,
    company_title: Optional[str] = None,
) -> List[Dict[str, Any]]:
    docling = doc.get("docling", {}) if isinstance(doc, dict) else {}
    tables = docling.get("tables", []) if isinstance(docling, dict) else []
    if not isinstance(tables, list):
        return []

    table_chunks: List[Dict[str, Any]] = []
    for t_idx, table_obj in enumerate(tables):
        if not isinstance(table_obj, dict):
            continue
        payload = _table_to_structured_payload(table_obj)
        headers = payload.get("headers", [])
        rows = payload.get("rows", [])
        if not headers and not rows:
            continue

        caption_refs = table_obj.get("captions", []) if isinstance(table_obj.get("captions"), list) else []
        caption = ""
        if caption_refs:
            caption = " | ".join(
                normalize_whitespace(x.get("text", "")) for x in caption_refs if isinstance(x, dict)
            ).strip()

        row_lines = []
        for r in rows[:120]:
            row_lines.append("; ".join(f"{k}: {v}" for k, v in r.items() if normalize_whitespace(v)))

        table_text_parts = [
            f"Structured Financial Table {t_idx + 1}",
            f"Caption: {caption}" if caption else "",
            f"Headers: {', '.join(headers)}" if headers else "",
            f"Dimensions: rows={payload.get('num_rows', 0)}, cols={payload.get('num_cols', 0)}",
            "Rows:",
            "\n".join(row_lines) if row_lines else "(no data rows)",
        ]
        table_text = "\n".join(part for part in table_text_parts if part).strip()

        table_text_l = table_text.lower()
        is_front_matter = False
        if str(form_type).upper() == "8-K":
            front_markers = [
                "state or other jurisdiction of incorporation",
                "commission file number",
                "i.r.s. employer identification no",
                "written communications pursuant to rule 425",
                "soliciting material pursuant to rule 14a-12",
                "pre-commencement communications pursuant to rule 14d-2",
                "pre-commencement communications pursuant to rule 13e-4",
                "title of each class",
                "trading symbol",
                "name of each exchange on which registered",
            ]
            is_front_matter = any(x in table_text_l for x in front_markers)

        metadata = {
            "chunk_id": str(uuid.uuid4()),
            "source": source_doc,
            "filename": filename,
            "ticker": ticker.upper(),
            "year": int(year),
            "form_type": form_type,
            "company_title": company_title,
            "content_type": "table",
            "table_index": t_idx,
            "table_caption": caption[:500],
            "table_headers": headers,
            "table_row_count": len(rows),
            "table_num_rows_raw": int(payload.get("num_rows", 0) or 0),
            "table_num_cols_raw": int(payload.get("num_cols", 0) or 0),
            "is_front_matter": is_front_matter,
        }
        table_chunks.append({"id": metadata["chunk_id"], "text": table_text, "metadata": metadata})

    logger.info("Extracted structured table chunks: %s", len(table_chunks))
    return table_chunks


# -----------------------------
# CHUNK BUILDING
# -----------------------------
# Helper function for 8-k filings
def is_8k_front_matter_text(text: str) -> bool:
    text = normalize_whitespace(text).lower()
    patterns = [
        "commission file number",
        "state or other jurisdiction of incorporation",
        "i.r.s. employer identification no",
        "title of each class",
        "trading symbol",
        "name of each exchange on which registered",
        "written communications pursuant to rule 425",
        "soliciting material pursuant to rule 14a-12",
        "pre-commencement communications pursuant to rule 14d-2",
        "pre-commencement communications pursuant to rule 13e-4",
    ]
    return any(p in text for p in patterns)

def semantic_blocks_to_chunks(
    blocks: List[Dict[str, Any]],
    source_doc: str,
    filename: str,
    ticker: str,
    year: int,
    form_type: str,
    company_title: Optional[str] = None
) -> List[Dict[str, Any]]:
    chunks = []

    for block_idx, block in enumerate(blocks):
        content = block["content"]
        part_title = block.get("part_title")
        item_title = block.get("item_title")
        note_title = block.get("note_title")
        subsection_title_raw = block.get("subsection_title_raw")
        subsection_title_clean = block.get("subsection_title_clean")
        subsection_title_normalized = block.get("subsection_title_normalized")

        is_front = False
        if str(form_type).upper() == "8-K":
            is_front = not item_title and is_8k_front_matter_text(content)
        else:
            is_front = block_idx < 3 and not (item_title or note_title)

        sub_chunks = chunk_words(
            content,
            max_words=MAX_CHUNK_WORDS,
            overlap_words=CHUNK_OVERLAP_WORDS
        )

        for local_chunk_idx, sub_chunk in enumerate(sub_chunks):
            chunk_text = build_chunk_text(
                part_title=part_title,
                item_title=item_title,
                note_title=note_title,
                subsection_title=subsection_title_normalized,
                content=sub_chunk
            )

            metadata = {
                "chunk_id": str(uuid.uuid4()),
                "source": source_doc,
                "filename": filename,
                "ticker": ticker.upper(),
                "year": int(year),
                "form_type": form_type,
                "company_title": company_title,
                "part_title": part_title,
                "item_title": item_title,
                "note_title": note_title,
                "subsection_title_raw": subsection_title_raw,
                "subsection_title_clean": subsection_title_clean,
                "subsection_title_normalized": subsection_title_normalized,
                "subsection_title": subsection_title_normalized,
                "block_index": block_idx,
                "chunk_index_within_block": local_chunk_idx,
                "content_type": "text",
                "is_front_matter": is_front,
            }

            chunks.append({
                "id": metadata["chunk_id"],
                "text": chunk_text,
                "metadata": metadata
            })

    logger.info("Generated chunks from semantic blocks: %s", len(chunks))
    return chunks


def is_low_value_chunk(metadata, text):
    
    if str(metadata.get("content_type", "")).lower() == "table":
        form_type = normalize_whitespace(metadata.get("form_type", "")).upper()
        text_l = text.lower()

        if form_type == "8-K":
            low_value_table_markers = [
                "state or other jurisdiction of incorporation",
                "commission file number",
                "i.r.s. employer identification no",
                "written communications pursuant to rule 425",
                "soliciting material pursuant to rule 14a-12",
                "pre-commencement communications pursuant to rule 14d-2",
                "pre-commencement communications pursuant to rule 13e-4",
                "title of each class",
                "trading symbol",
                "name of each exchange on which registered",
            ]
            if any(x in text_l for x in low_value_table_markers):
                return True

        return False

    if metadata.get("note_title") is None and metadata.get("item_title") is None:
        return True

    text_l = text.lower()
    company_title = normalize_whitespace(metadata.get("company_title", "")).lower()
    form_type = normalize_whitespace(metadata.get("form_type", "")).upper()

    bad_starts = [
        "for the quarterly period ended",
        "for the fiscal year ended",
        "securities registered pursuant",
    ]
    if company_title:
        bad_starts.append(company_title)
    if form_type:
        bad_starts.append(f"form {form_type.lower()}")
    if any(text_l.startswith(x) for x in bad_starts):
        return True

    return False


def should_keep_chunk(chunk):
    md = chunk["metadata"]
    text = chunk["text"].strip()
    words = len(text.split())
    # if str(md.get("content_type", "")).lower() == "table":
    #     return bool(md.get("table_headers")) or int(md.get("table_row_count", 0) or 0) > 0 or words >= 20
    if str(md.get("content_type", "")).lower() == "table":
        form_type = normalize_whitespace(md.get("form_type", "")).upper()
        text_l = text.lower()

        if form_type == "8-K":
            useful_markers = [
                "exhibit",
                "financial statements",
                "results of operations",
                "earnings",
                "revenue",
                "net income",
                "diluted earnings per share",
                "balance sheet",
                "cash flow",
                "99.1",
                "99.2",
            ]
            if not any(x in text_l for x in useful_markers):
                return False

        return bool(md.get("table_headers")) or int(md.get("table_row_count", 0) or 0) > 0 or words >= 20

    has_item = bool(normalize_whitespace(md.get("item_title", "")))
    has_note = bool(normalize_whitespace(md.get("note_title", "")))
    has_subsection = bool(normalize_whitespace(md.get("subsection_title", "")))
    has_structural_context = has_item or has_note or has_subsection

    if words < 40:
        return False

    # Keep chunk if it has structure (item/note/subsection) OR enough body text.
    # This is intentionally relaxed for useful narrative content that may lack
    # explicit item/note labels after normalization.
    if not has_structural_context and words < MIN_CHUNK_WORDS:
        return False

    if text.lower().startswith("for the quarterly period ended"):
        return False

    return True


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def docling_json_to_pinecone_chunks(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_doc = doc.get("source", "")
    ticker = doc.get("ticker", "")
    year = doc.get("year", "")
    company_title = doc.get("company_title", "")
    form_type = doc.get("form_type", "")
    filename = doc.get("docling", {}).get("origin", {}).get("filename", source_doc)

    nodes = extract_text_nodes(doc)
    cleaned_nodes = clean_text_nodes(
        nodes,
        company_title=company_title,
        form_type=form_type,
    )
    blocks = build_semantic_blocks(cleaned_nodes)
    chunks = semantic_blocks_to_chunks(
        blocks=blocks,
        source_doc=source_doc,
        filename=filename,
        ticker=ticker,
        year=year,
        form_type=form_type,
        company_title=company_title
    )
    if str(form_type).upper() == "8-K":
        logging.info("\nDetected semantic blocks for 8-K:")
        for i, b in enumerate(blocks[:20]):
            logger.info(
                "%s | item: %s | subsection: %s",
                i,
                b.get("item_title"),
                b.get("subsection_title_normalized"),
            )
            logger.info("%s", b.get("content", "")[:200])
            logger.info("%s", "-" * 80)
    table_chunks = extract_structured_table_chunks(
        doc=doc,
        source_doc=source_doc,
        filename=filename,
        ticker=ticker,
        year=year,
        form_type=form_type,
        company_title=company_title,
    )
    chunks.extend(table_chunks)
    logger.info(
        "Doc processed: source=%s ticker=%s year=%s form_type=%s chunks=%s",
        source_doc,
        ticker,
        year,
        form_type,
        len(chunks),
    )
    return chunks


# -----------------------------
# SAVE OUTPUT
# -----------------------------
def save_chunks_to_json(chunks: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info("Saved chunks JSON: path=%s count=%s", output_path, len(chunks))


def print_detected_subsections(chunks):
    seen = set()
    for ch in chunks:
        sub = ch["metadata"].get("subsection_title")
        if sub and sub not in seen:
            seen.add(sub)
            logger.info("%s", sub)


def print_rejected_subsections(chunks):
    all_subsections = set()
    for ch in chunks:
        raw_sub = ch.get("metadata", {}).get("subsection_title", "")
        normalized_sub = normalize_whitespace(raw_sub)
        if normalized_sub:
            all_subsections.add(normalized_sub)
    rejected = 0
    for sub in sorted(all_subsections):
        if is_bad_final_subsection_title(sub):
            logger.info("REJECTED: %s", sub)
            rejected += 1
    if rejected == 0:
        logger.info("REJECTED: (none)")



