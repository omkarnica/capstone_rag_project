"""
Extracting 10-k, 10-q,8-k files from sec.gov

    End-to-end ingestion flow:
    1) Download filings
    2) Extract/clean XBRL
    3) Convert to Docling JSON
    4) Verify/validate Docling output
    5) Validate XBRL output

"""


import requests
import pandas as pd
import time
import os
import json
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings
import re
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from dotenv import load_dotenv
from config_loader import load_config_yaml


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
_CONFIG = load_config_yaml(BASE_DIR / "config.yaml")


def _cfg_str(key: str, default: str) -> str:
    raw = str(_CONFIG.get(key, default)).strip()
    if "=" in raw:
        raw = raw.split("=", 1)[1].strip()
    return raw.strip('"').strip("'")


SEC_USER_AGENT_ENV = _cfg_str("SEC_USER_AGENT_ENV", "SEC_USER_AGENT")
SEC_USER_AGENT = os.getenv(
    SEC_USER_AGENT_ENV,
    "FinancialAppProject(contact@example.com)",
).strip()

HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger("ingestion_filings")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


def _inserted_at() -> str:
    return datetime.now().isoformat(timespec="seconds")


LIFECYCLE_LOG_FILE = os.path.join(
    os.path.dirname(__file__),
    "document_lifecycle_log.jsonl",
)


def record_document_lifecycle(file_path, stage, ticker=None, year=None, form_type=None):
    event = {
        "file": os.path.abspath(str(file_path)),
        "ingested_at": _inserted_at(),
        "stage": str(stage),
        "ticker": str(ticker) if ticker else None,
        "year": int(year) if str(year).isdigit() else year,
        "form_type": str(form_type) if form_type else None,
    }
    event = {k: v for k, v in event.items() if v is not None}
    try:
        with open(LIFECYCLE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.exception("Failed to write lifecycle event for %s: %s", file_path, e)


# -------------------------------
# Scrape S&P 500 Ticker List & get ticker, cik, and title table
# -------------------------------

# Unified function to get ticker, cik, and title as a DataFrame
def get_ticker_cik_table():
    logger.info("Entering get_ticker_cik_table")
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        logger.error("Error fetching %s: %s", url, response.status_code)
        logger.info("Response preview: %s", response.text[:500])
        return None
    try:
        data = response.json()
    except Exception as e:
        logger.exception("Error parsing JSON from %s: %s", url, e)
        logger.info("Response preview: %s", response.text[:500])
        return None
    records = [
        {
            "ticker": v["ticker"],
            "cik": str(v["cik_str"]).zfill(10),
            "title": v["title"]
        }
        for v in data.values()
    ]
    df = pd.DataFrame(records)
    logger.info("Successfully retrieved ticker, cik, and title table.")
    logger.info("Ticker table preview:\n%s", df.head().to_string(index=False))
    return df



# -------------------------------
# Get Filing Manifest (Accession Number)
# -------------------------------
def get_filings(cik):
    logger.info("Entering get_filings")
    headers = {
        "User-Agent": SEC_USER_AGENT,
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'data.sec.gov'
    }
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error("Error fetching %s: %s", url, response.status_code)
        return []
    try:
        data = response.json()
    except Exception as e:
        logger.exception("Error parsing JSON from %s: %s", url, e)
        return []

    if 'filings' not in data or 'recent' not in data['filings']:
        logger.error("Missing 'filings' or 'recent' in data from %s", url)
        return []

    filings = data['filings']['recent']
    results = []
    for i, form in enumerate(filings['form']):
        # Accept any form type (e.g., 10-K, 10-Q) in the pipeline
        results.append({
            "form": form,
            "year": int(filings['filingDate'][i][:4]) if 'filingDate' in filings and filings['filingDate'][i] else None,
            "accession": filings['accessionNumber'][i].replace('-', ''),
            "primary_doc": filings['primaryDocument'][i],
            "report_date": filings.get('reportDate', [None]*len(filings['form']))[i],
            "filing_url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{filings['accessionNumber'][i].replace('-', '')}/{filings['primaryDocument'][i]}"
        })
    return results

# -------------------------------
# Ingestion Pipeline
# -------------------------------
def _normalize_company_name(name):
    logger.info("Entering _normalize_company_name")
    if not name:
        return ""
    # Keep alphanumerics and spaces only
    return re.sub(r"[^a-z0-9 ]+", "", name.lower()).strip()


def ingestion_pipeline(company_title, form_type="10-K", ticker_override=None):
    logger.info("Entering ingestion_pipeline")
    ticker_cik_df = get_ticker_cik_table()
    row = pd.DataFrame()

    if ticker_override:
        row = ticker_cik_df[
            ticker_cik_df["ticker"].str.upper() == str(ticker_override).upper()
        ]

    if row.empty and company_title:
        norm_target = _normalize_company_name(company_title)
        titles = ticker_cik_df["title"].astype(str)
        norm_titles = titles.apply(_normalize_company_name)

        # Exact normalized match
        row = ticker_cik_df[norm_titles == norm_target]

        # Fuzzy-ish fallback: contains match
        if row.empty and norm_target:
            row = ticker_cik_df[norm_titles.str.contains(norm_target)]
    if not row.empty:
        cik = str(row.iloc[0]["cik"]).zfill(10)
        ticker = row.iloc[0]["ticker"]
    else:
        logger.error("Company '%s' not found in table.", company_title)
        return None

    filings = get_filings(cik)
    if not filings:
        logger.error("No filings found for %s.", company_title)
        return None

    # Filter by form type (e.g., 10-K, 10-Q) and by year (2021-2025)
    filtered_filings = [
        f for f in filings
        if f["form"] == form_type and f["year"] is not None and 2021 <= f["year"] <= 2025
    ]
    if not filtered_filings:
        logger.error("No %s filings from 2021 to 2025 found for %s.", form_type, company_title)
        return None

    folder = f"{ticker.lower()}_{form_type.lower()}"
    os.makedirs(folder, exist_ok=True)

    for filing in filtered_filings:
        url = filing['filing_url']
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code == 200:
                filename = f"{folder}/{filing['year']}.html"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(response.text)
                logger.info("Saved filing HTML: %s | inserted_at=%s", filename, _inserted_at())
                record_document_lifecycle(
                    file_path=filename,
                    stage="raw_html_downloaded",
                    ticker=ticker,
                    year=filing.get("year"),
                    form_type=form_type,
                )
            time.sleep(1.5)
        except Exception as e:
            logger.exception("Error downloading %s: %s", url, e)
    return folder, ticker

# -------------------------------
# XBRL Extraction and Cleaning
# -------------------------------

def normalize_value(val):
    logger.info("Entering normalize_value")
    if val is None:
        return None
    try:
        return float(str(val).replace(",", ""))
    except:
        return val


def is_text_block(tag):
    logger.info("Entering is_text_block")
    return tag and "TextBlock" in tag


def parse_segment(segment):
    logger.info("Entering parse_segment")
    if isinstance(segment, list) and segment:
        seg = segment[0]
    elif isinstance(segment, dict):
        seg = segment
    else:
        return None, None

    dim = seg.get("dimension")
    member = seg.get("value")
    return dim, member


def compute_period_info(start_date, end_date, period_focus=None):
    logger.info("Entering compute_period_info")
    if period_focus:
        period_label = str(period_focus).upper()
    else:
        period_label = None

    period_type = None

    try:
        if start_date and end_date:
            from datetime import datetime

            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            days = (end_dt - start_dt).days

            if days <= 110:
                period_type = "quarterly"
                if not period_label:
                    q = ((end_dt.month - 1) // 3) + 1
                    period_label = f"Q{q}"
            elif days <= 200:
                period_type = "semiannual"
                if not period_label:
                    period_label = "H1" if end_dt.month <= 6 else "H2"
            elif days <= 400:
                period_type = "annual"
                if not period_label:
                    period_label = "FY"
            else:
                period_type = "other"
                if not period_label:
                    period_label = "TTM"
    except Exception:
        # Leave period_label/period_type as-is on parsing errors
        pass

    return period_label, period_type


def clean_xbrl_data(data):
    logger.info("Entering clean_xbrl_data")
    cleaned_data = []
    dimension_mappings = {
        "CloudComputingMember": "Azure Cloud",
        "ProductDivisionAxis": "Product Division",
        "GeographyAxis": "Geographic Region"
    }

    for filing in data:
        # Pull filing-level metadata from facts where available
        tag_values = {}
        for fact in filing.get("facts", []):
            tag = fact.get("tag") or fact.get("name")
            val = fact.get("value")
            if tag and val and tag not in tag_values:
                tag_values[tag] = val

        company_name = tag_values.get("dei:EntityRegistrantName")
        cik = tag_values.get("dei:EntityCentralIndexKey")
        filing_type = tag_values.get("dei:DocumentType")
        fiscal_year = tag_values.get("dei:DocumentFiscalYearFocus") or filing.get("year")
        period_focus = tag_values.get("dei:DocumentFiscalPeriodFocus")

        seen = set()
        cleaned_facts = []

        for fact in filing.get("facts", []):
            tag = fact.get("tag") or fact.get("name")

            # -----------------------------
            # REMOVE TEXT BLOCKS
            # -----------------------------
            if is_text_block(tag):
                continue

            value = normalize_value(fact.get("value"))

            # -----------------------------
            # VALIDATE + CLEAN FACTS
            # -----------------------------
            if value is None:
                continue

            if tag and tag.startswith("us-gaap:") and not isinstance(value, (int, float)):
                continue

            # -----------------------------
            # FIX SEGMENTS (ensure list)
            # -----------------------------
            segment = fact.get("segment")

            if segment:
                if isinstance(segment, dict):
                    segment = [segment]  # convert to list
                elif not isinstance(segment, list):
                    segment = None

            # -----------------------------
            # FIX CONTEXT (ensure valid)
            # -----------------------------
            start_date = fact.get("start_date")
            end_date = fact.get("end_date")
            instant = fact.get("instant")

            # sanity check: must have either instant OR duration
            if not (instant or (start_date and end_date)):
                continue  # skip invalid facts

            # -----------------------------
            # DEDUPLICATION
            # -----------------------------
            unique_key = (
                tag,
                fact.get("context"),
                start_date,
                end_date,
                instant
            )

            if unique_key in seen:
                continue

            seen.add(unique_key)

            # -----------------------------
            #  FINAL CLEANED FACT
            # -----------------------------
            # Default unit for us-gaap numeric facts
            unit = fact.get("unit")
            if tag and tag.startswith("us-gaap:") and not unit:
                unit = "iso4217:USD"

            period_label, period_type = compute_period_info(
                start_date,
                end_date,
                period_focus=period_focus,
            )

            fiscal_quarter = None
            if period_label and str(period_label).upper().startswith("Q"):
                fiscal_quarter = str(period_label).upper()

            dim, member = parse_segment(segment)
            cleaned_fact = {
                "tag": tag,
                "name": fact.get("name") or tag,
                "value": value,
                "unit": unit,
                "context": fact.get("context"),
                "year": fact.get("year"),
                "fiscal_year": fiscal_year,
                "fiscal_quarter": fiscal_quarter,
                "period_label": period_label,
                "period_type": period_type,
                "start_date": start_date,
                "end_date": end_date,
                "instant": instant,
                "segment": segment,
                "dimension": dimension_mappings.get(dim, dim),
                "dimension_member": dimension_mappings.get(member, member)
            }

            cleaned_facts.append(cleaned_fact)

        cleaned_data.append({
            "source": filing.get("source"),
            "ticker": filing.get("ticker"),
            "year": filing.get("year"),
            "company_name": company_name,
            "cik": cik,
            "filing_type": filing_type,
            "facts": cleaned_facts
        })

    return cleaned_data


def extract_xbrl_from_filings(folder):
    logger.info("Entering extract_xbrl_from_filings")
    xbrl_data = []
    ticker = folder.split('_')[0]

    for filename in os.listdir(folder):
        if not filename.endswith(".html"):
            continue

        file_path = os.path.join(folder, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "lxml")
        except Exception as e:
            logger.exception("Error reading HTML file %s: %s", file_path, e)
            continue

        # -----------------------------
        # STEP 1: Extract CONTEXTS
        # -----------------------------
        context_map = {}

        for ctx in soup.find_all(lambda tag: tag.name and tag.name.endswith("context")):
            ctx_id = ctx.get("id")

            if not ctx_id:
                continue

            start_date = None
            end_date = None
            instant = None
            segment = None

            period = ctx.find(lambda t: t.name and t.name.endswith("period"))

            if period:
                start = period.find(lambda t: t.name and t.name.endswith("startdate"))
                end = period.find(lambda t: t.name and t.name.endswith("enddate"))
                inst = period.find(lambda t: t.name and t.name.endswith("instant"))

                if start:
                    start_date = start.text.strip()
                if end:
                    end_date = end.text.strip()
                if inst:
                    instant = inst.text.strip()

            # Extract segment (dimension)
            seg = ctx.find(lambda t: t.name and t.name.endswith("explicitmember"))
            if seg:
                segment = {
                    "dimension": seg.get("dimension"),
                    "value": seg.text.strip()
                }

            context_map[ctx_id] = {
                "start_date": start_date,
                "end_date": end_date,
                "instant": instant,
                "segment": segment
            }

        # -----------------------------
        # STEP 2: Extract UNITS
        # -----------------------------
        unit_map = {}

        for unit in soup.find_all(lambda tag: tag.name and tag.name.endswith("unit")):
            unit_id = unit.get("id")

            measure = unit.find(lambda t: t.name and t.name.endswith("measure"))

            if unit_id and measure:
                unit_map[unit_id] = measure.text.split(":")[-1]

        # -----------------------------
        # ✅ STEP 3: Extract FACTS
        # -----------------------------
        facts = []

        year = filename.replace(".html", "")

        fact_tags = soup.find_all(
            lambda tag: tag.name and (
                tag.name.endswith("nonfraction") or tag.name.endswith("nonnumeric")
            )
        )

        for tag in fact_tags:
            # HTML parsers often lowercase attribute names
            name = tag.get("name") or tag.get("name".lower())
            context_ref = tag.get("contextref") or tag.get("contextRef")
            unit_ref = tag.get("unitref") or tag.get("unitRef")

            # Handle nil values
            if tag.get("xsi:nil") == "true":
                value = None
            else:
                value = tag.text.strip()

            context_info = context_map.get(context_ref, {})
            unit_value = unit_map.get(unit_ref)

            fact = {
                "name": name,
                "tag": name,
                "value": value,
                "unit": unit_value,
                "context": context_ref,
                "year": int(year) if year.isdigit() else year,
                "start_date": context_info.get("start_date"),
                "end_date": context_info.get("end_date"),
                "instant": context_info.get("instant"),
                "segment": context_info.get("segment")
            }

            facts.append(fact)

        # -----------------------------
        # STEP 4: FILE METADATA
        # -----------------------------
        source_name = f"{ticker}_{year}.html"

        xbrl_data.append({
            "source": source_name,
            "ticker": ticker,
            "year": int(year) if year.isdigit() else year,
            "facts": facts
        })

    # -----------------------------
    # STEP 5: SAVE JSON
    # -----------------------------
    company_filename = f"{folder}.json"

    if os.path.exists(company_filename):
        with open(company_filename, "r", encoding="utf-8") as json_file:
            try:
                existing_data = json.load(json_file)
            except Exception:
                logger.exception("Error reading JSON file %s", company_filename)
                existing_data = []
    else:
        existing_data = []

    existing_by_source = {
        entry["source"]: entry
        for entry in existing_data
        if "source" in entry
    }

    for entry in xbrl_data:
        src = entry.get("source")
        if not src:
            continue

        if src in existing_by_source:
            existing_by_source[src]["facts"] = entry.get("facts", [])
        else:
            existing_data.append(entry)

    cleaned_data = clean_xbrl_data(existing_data)
    with open(company_filename, "w", encoding="utf-8") as json_file:
        json.dump(cleaned_data, json_file, ensure_ascii=False, indent=2)
    logger.info("Saved cleaned XBRL JSON: %s | inserted_at=%s", company_filename, _inserted_at())
    record_document_lifecycle(
        file_path=company_filename,
        stage="xbrl_cleaned_json_saved",
        ticker=ticker,
    )




# -------------------------------
# XBRL data removal from HTML and Docling conversion
# -------------------------------

def _strip_inline_xbrl(html_text: str) -> str:
    logger.info("Entering _strip_inline_xbrl")
    soup = BeautifulSoup(html_text, "lxml")

    # Remove non-content ixbrl containers entirely
    for tag in soup.find_all(True):
        name = (tag.name or "").lower()
        if ":" in name:
            prefix, local = name.split(":", 1)
        else:
            prefix, local = "", name

        if prefix == "ix" and local in {"header", "hidden", "resources", "references"}:
            tag.decompose()

    # Unwrap inline ixbrl tags so visible text remains
    for tag in soup.find_all(True):
        name = (tag.name or "").lower()
        if ":" in name:
            prefix, local = name.split(":", 1)
        else:
            prefix, local = "", name

        # Inline XBRL wrappers around visible numbers/text
        if prefix == "ix" and local in {"nonfraction", "nonnumeric"}:
            tag.unwrap()
            continue

        # Other XBRL namespace tags: unwrap to keep any visible text
        if prefix in {
            "xbrli",
            "xbrldi",
            "link",
            "xlink",
            "dei",
            "us-gaap",
            "srt",
            "ixt",
            "ixt-sec",
            "iso4217",
        }:
            tag.unwrap()

    return str(soup)


def extract_docling_from_filings(folder, company_title=None, form_type=None, write_cleaned_html=False):
    logger.info("Entering extract_docling_from_filings")
    docling_data = []
    ticker = folder.split("_")[0]
    if form_type is None and "_" in folder:
        form_type = folder.split("_", 1)[1].replace("-", "-").upper()
    converter = DocumentConverter()

    for filename in os.listdir(folder):
        if filename.endswith(".html"):
            file_path = os.path.join(folder, filename)
            year = filename.replace(".html", "")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_html = f.read()
            except Exception as e:
                logger.exception("Error reading HTML file %s: %s", file_path, e)
                continue

            cleaned_html = _strip_inline_xbrl(raw_html)

            if write_cleaned_html:
                cleaned_path = os.path.join(folder, filename.replace(".html", ".cleaned.html"))
                try:
                    with open(cleaned_path, "w", encoding="utf-8") as cf:
                        cf.write(cleaned_html)
                    record_document_lifecycle(
                        file_path=cleaned_path,
                        stage="cleaned_html_saved",
                        ticker=ticker,
                        year=year,
                        form_type=form_type,
                    )
                except Exception as e:
                    logger.exception("Error writing cleaned HTML file %s: %s", cleaned_path, e)

            # Convert cleaned HTML to Docling document (string-based conversion)
            conv_res = converter.convert_string(
                content=cleaned_html,
                format=InputFormat.HTML,
                name=filename,
            )

            doc_dict = conv_res.document.export_to_dict(mode="json")

            source_name = f"{ticker}_{year}.html"
            docling_data.append(
                {
                    "source": source_name,
                    "ticker": ticker,
                    "year": int(year) if year.isdigit() else year,
                    "company_title": company_title,
                    "form_type": form_type,
                    "docling": doc_dict,
                }
            )

    # Save the result as JSON under the company filename
    company_filename = f"{folder}_docling.json"
    try:
        with open(company_filename, "w", encoding="utf-8") as json_file:
            json.dump(docling_data, json_file, ensure_ascii=False, indent=2)
        logger.info("Saved Docling JSON: %s | inserted_at=%s", company_filename, _inserted_at())
        record_document_lifecycle(
            file_path=company_filename,
            stage="docling_json_saved",
            ticker=ticker,
            form_type=form_type,
        )
    except Exception as e:
        logger.exception("Error writing Docling JSON file %s: %s", company_filename, e)


def verify_docling_output(json_path):
    logger.info("Entering verify_docling_output")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sources = [d.get("source") for d in data]
    unique_sources = sorted({s for s in sources if s})
    logger.info("Docling entries: %s", len(data))
    logger.info("Unique sources: %s", len(unique_sources))
    for s in unique_sources:
        logger.info("Docling source: %s", s)


def validate_docling_content(json_path, folder, expected_sections=None):
    logger.info("Entering validate_docling_content")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_sections = expected_sections or []

    for entry in data:
        source = entry.get("source")
        doc = entry.get("docling", {})
        texts = doc.get("texts", [])
        tables = doc.get("tables", [])

        chunks = [{"type": "text", "text": t.get("text", "")} for t in texts]

        # Expected sections check
        for section in expected_sections:
            if not any(section.lower() in chunk["text"].lower() for chunk in chunks):
                logger.info("[%s] Missing section: %s", source, section)

        # Table count check
        html_tables = None
        if source:
            year = source.split("_", 1)[-1].replace(".html", "")
            html_path = os.path.join(folder, f"{year}.html")
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as hf:
                    html = hf.read()
                html_tables = len(BeautifulSoup(html, "lxml").find_all("table"))
        parsed_tables = len(tables)

        if html_tables is not None:
            logger.info(f"[{source}] tables in HTML: {html_tables}, tables parsed: {parsed_tables}")
        else:
            logger.info(f"[{source}] tables parsed: {parsed_tables}")


def validate_xbrl_output(json_path):
    logger.info("Entering validate_xbrl_output")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    important_tags = [
        "us-gaap:Revenue",
        "us-gaap:NetIncomeLoss",
        "us-gaap:Assets",
    ]

    for entry in data:
        source = entry.get("source")
        facts = entry.get("facts", [])
        fact_names = {
            f.get("name") or f.get("tag")
            for f in facts
            if f.get("name") or f.get("tag")
        }

        for tag in important_tags:
            if tag not in fact_names:
                logger.info("[%s] Missing tag: %s", source, tag)

        for fact in facts:
            if not fact.get("context"):
                logger.info("[%s] Missing context: %s", source, fact)


def ingestion_filing(
    company_title,
    form_type="10-K",
    ticker_override=None,
    expected_sections=None,
):
    """
    End-to-end ingestion flow:
    1) Download filings
    2) Extract/clean XBRL
    3) Convert to Docling JSON
    4) Verify/validate Docling output
    5) Validate XBRL output
    """
    logger.info("Starting ingestion_filing for company_title=%s form_type=%s", company_title, form_type)

    result = ingestion_pipeline(
        company_title,
        form_type=form_type,
        ticker_override=ticker_override,
    )

    if not result:
        logger.error("Ingestion skipped: company lookup or filing download failed for %s", company_title)
        return None

    folder, ticker = result
    docling_json_path = f"{folder}_docling.json"
    xbrl_json_path = f"{folder}.json"

    extract_xbrl_from_filings(folder)
    extract_docling_from_filings(folder, company_title=company_title, form_type=form_type)
    verify_docling_output(docling_json_path)
    validate_docling_content(
        docling_json_path,
        folder=folder,
        expected_sections=expected_sections or ["Item 1", "Item 2", "Financial Statements"],
    )
    validate_xbrl_output(xbrl_json_path)

    logger.info(
        "Completed ingestion_filing: company_title=%s ticker=%s folder=%s",
        company_title,
        ticker,
        folder,
    )
    logging.info("XBRL extraction completed and saved to JSON!")

    return {
        "folder": folder,
        "ticker": ticker,
        "docling_json_path": docling_json_path,
        "xbrl_json_path": xbrl_json_path,
    }



