"""
SEC filings ingestion for the filings build pipeline.

Calling module:

    src.filings.pipeline.run_filings_pipeline()
      |
      v
    ingestion_filing()

Ingestion flow:

    ingestion_filing()
      |
      |-- ingestion_pipeline() downloads filing HTML from sec.gov
      |-- extract_xbrl_from_filings() writes structured XBRL JSON
      |-- extract_docling_from_filings() writes Docling JSON
      |-- verify_docling_output()
      |-- validate_docling_content()
      `-- validate_xbrl_output()

Primary output:

    {"folder", "docling_json_path", "xbrl_json_path", "company_title",
     "form_type", "ticker"}
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
from .config_loader import load_config_yaml


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
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
        filing_date = filings.get('filingDate', [None] * len(filings['form']))[i]
        report_date = filings.get('reportDate', [None] * len(filings['form']))[i]
        year_source = report_date or filing_date
        # Accept any form type (e.g., 10-K, 10-Q) in the pipeline
        results.append({
            "form": form,
            "year": int(year_source[:4]) if year_source else None,
            "filing_date": filing_date,
            "accession": filings['accessionNumber'][i].replace('-', ''),
            "primary_doc": filings['primaryDocument'][i],
            "report_date": report_date,
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

    folder = str(DATA_DIR / f"{ticker.lower()}_{form_type.lower()}")
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
    return folder, ticker, cik, filtered_filings


def _find_exhibit_21_filename(index_json):
    logger.info("Entering _find_exhibit_21_filename")
    items = index_json.get("directory", {}).get("item", [])
    for item in items:
        item_type = str(item.get("type", "")).upper()
        item_name = str(item.get("name", "")).lower()
        if item_type.startswith("EX-21") or "exhibit21" in item_name or "ex21" in item_name:
            return item.get("name")
    return None


def _lookup_exhibit_21_filename(cik, accession_nodash):
    logger.info("Entering _lookup_exhibit_21_filename")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession_nodash}/index.json"
    )
    try:
        response = requests.get(index_url, headers=HEADERS)
        if response.status_code != 200:
            logger.info("No filing index JSON found for %s: status=%s", index_url, response.status_code)
            return None
        return _find_exhibit_21_filename(response.json())
    except Exception as e:
        logger.exception("Error fetching filing index %s: %s", index_url, e)
        return None


def _build_exhibit_21_url(cik, accession_nodash, exhibit_filename):
    logger.info("Entering _build_exhibit_21_url")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession_nodash}/{exhibit_filename}"
    )


def _fetch_exhibit_21_html(cik, accession_nodash, exhibit_filename):
    logger.info("Entering _fetch_exhibit_21_html")
    exhibit_url = _build_exhibit_21_url(cik, accession_nodash, exhibit_filename)
    try:
        response = requests.get(exhibit_url, headers=HEADERS)
        if response.status_code != 200:
            logger.info("Failed to fetch Exhibit 21 %s: status=%s", exhibit_url, response.status_code)
            return None
        return response.text
    except Exception as e:
        logger.exception("Error downloading Exhibit 21 %s: %s", exhibit_url, e)
        return None


def _normalize_subsidiary_name(text):
    logger.info("Entering _normalize_subsidiary_name")
    return re.sub(r"\s+", " ", str(text or "")).strip(" -:\t\r\n")


def _looks_like_header_row(text):
    logger.info("Entering _looks_like_header_row")
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "name of subsidiary",
            "subsidiaries",
            "jurisdiction",
            "state of incorporation",
            "exhibit 21",
            "list of subsidiaries",
        )
    )


def _extract_subsidiaries_from_exhibit_html(html_text):
    logger.info("Entering _extract_subsidiaries_from_exhibit_html")
    if not html_text:
        return []

    soup = BeautifulSoup(html_text, "lxml")
    subsidiaries = []
    seen = set()

    def _add_candidate(candidate):
        name = _normalize_subsidiary_name(candidate)
        if not name:
            return
        if _looks_like_header_row(name):
            return
        if len(name) < 3 or len(name) > 160:
            return
        if not re.search(r"[A-Za-z]", name):
            return
        if name.lower() in seen:
            return
        seen.add(name.lower())
        subsidiaries.append(name)

    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            first_cell = _normalize_subsidiary_name(cells[0].get_text(" ", strip=True))
            if row.find("td"):
                _add_candidate(first_cell)

    if subsidiaries:
        return subsidiaries

    for item in soup.find_all(["li", "p"]):
        text = _normalize_subsidiary_name(item.get_text(" ", strip=True))
        _add_candidate(text)

    return subsidiaries


def collect_subsidiaries_by_year(
    cik,
    ticker,
    company_title,
    form_type,
    filings,
    data_dir=DATA_DIR,
):
    logger.info("Entering collect_subsidiaries_by_year")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    subsidiaries_by_year = []
    for filing in filings:
        accession = filing.get("accession")
        year = filing.get("year")
        if not accession or not year:
            continue

        exhibit_filename = _lookup_exhibit_21_filename(cik, accession)
        if not exhibit_filename:
            logger.info("No Exhibit 21 found for ticker=%s year=%s accession=%s", ticker, year, accession)
            continue

        exhibit_url = _build_exhibit_21_url(cik, accession, exhibit_filename)
        exhibit_html = _fetch_exhibit_21_html(cik, accession, exhibit_filename)
        if not exhibit_html:
            continue

        subsidiaries = _extract_subsidiaries_from_exhibit_html(exhibit_html)
        subsidiaries_by_year.append(
            {
                "year": int(year) if str(year).isdigit() else year,
                "accession": accession,
                "exhibit_21_url": exhibit_url,
                "subsidiaries": subsidiaries,
            }
        )

    output_path = data_dir / f"{ticker.lower()}_{form_type.lower()}_subsidiaries.json"
    payload = {
        "ticker": ticker,
        "company_title": company_title,
        "form_type": form_type,
        "subsidiaries_by_year": subsidiaries_by_year,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    record_document_lifecycle(
        file_path=output_path,
        stage="subsidiaries_json_saved",
        ticker=ticker,
        form_type=form_type,
    )
    return output_path


def _filter_def14a_filings(filings):
    logger.info("Entering _filter_def14a_filings")
    return [
        filing for filing in filings
        if filing.get("form") == "DEF 14A"
        and filing.get("year") is not None
        and 2021 <= filing["year"] <= 2025
    ]


def _fetch_proxy_filing_html(filing_url):
    logger.info("Entering _fetch_proxy_filing_html")
    try:
        response = requests.get(filing_url, headers=HEADERS)
        if response.status_code != 200:
            logger.info("Failed to fetch proxy filing %s: status=%s", filing_url, response.status_code)
            return None
        return response.text
    except Exception as e:
        logger.exception("Error downloading proxy filing %s: %s", filing_url, e)
        return None


def _normalize_board_member_text(text):
    logger.info("Entering _normalize_board_member_text")
    return re.sub(r"\s+", " ", str(text or "")).strip(" -:\t\r\n")


def _extract_table_headers(table):
    logger.info("Entering _extract_table_headers")
    header_cells = table.find_all("th")
    if header_cells:
        return [
            _normalize_board_member_text(cell.get_text(" ", strip=True)).lower()
            for cell in header_cells
        ]

    first_row = table.find("tr")
    if not first_row:
        return []
    return [
        _normalize_board_member_text(cell.get_text(" ", strip=True)).lower()
        for cell in first_row.find_all("td")
    ]


def _get_table_context_text(table):
    logger.info("Entering _get_table_context_text")
    context_parts = []
    if table.find("caption"):
        context_parts.append(table.find("caption").get_text(" ", strip=True))

    sibling = table.find_previous(["h1", "h2", "h3", "h4", "strong", "b", "p"])
    if sibling:
        context_parts.append(sibling.get_text(" ", strip=True))

    return " ".join(context_parts).lower()


def _looks_like_director_table(headers, context_text):
    logger.info("Entering _looks_like_director_table")
    header_text = " ".join(headers)
    combined_text = f"{header_text} {context_text}"

    has_name_column = any(
        keyword in header_text
        for keyword in ("name", "director", "nominee")
    )
    has_role_column = any(
        keyword in header_text
        for keyword in (
            "occupation",
            "position",
            "title",
            "principal",
            "experience",
            "business",
        )
    )
    director_context = any(
        keyword in combined_text
        for keyword in ("board of directors", "nominees for election", "director nominees", "election of directors")
    )
    fee_or_numeric_table = any(
        keyword in combined_text
        for keyword in (
            "fees earned",
            "shares of common stock",
            "beneficially owned",
            "cash",
            "award type",
            "grant date",
            "value realized",
            "percentile",
            "vote required",
            "executive officers",
            "executive compensation",
            "security ownership",
            "equity compensation",
            "management proposals",
        )
    )
    looks_like_year_matrix = "year" in header_text and not director_context
    return has_name_column and (has_role_column or director_context) and not fee_or_numeric_table and not looks_like_year_matrix


def _find_column_index(headers, keywords, default=0):
    logger.info("Entering _find_column_index")
    for index, header in enumerate(headers):
        if any(keyword in header for keyword in keywords):
            return index
    return default


def _is_person_like_name(name):
    logger.info("Entering _is_person_like_name")
    clean_name = _normalize_board_member_text(name)
    if not clean_name:
        return False
    if any(char.isdigit() for char in clean_name):
        return False
    if "http" in clean_name.lower():
        return False
    disallowed_phrases = (
        "proxy statement",
        "report",
        "committee",
        "proposal",
        "meeting",
        "shareholder",
        "percentile",
        "fees",
        "general information",
        "business highlights",
        "security ownership",
        "equity compensation",
        "annual board",
    )
    if any(phrase in clean_name.lower() for phrase in disallowed_phrases):
        return False

    tokens = re.findall(r"[A-Za-z&+'().-]+", clean_name)
    alpha_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
    if len(alpha_tokens) < 2 or len(alpha_tokens) > 6:
        return False

    capitalized_tokens = [
        token for token in alpha_tokens
        if token[0].isupper() or token.lower() in {"de", "la", "van", "von"}
    ]
    return len(capitalized_tokens) >= 2


def _is_board_role_like(title):
    logger.info("Entering _is_board_role_like")
    clean_title = _normalize_board_member_text(title)
    if not clean_title:
        return False
    if re.fullmatch(r"[\d,.%$() -]+", clean_title):
        return False
    if "http" in clean_title.lower():
        return False
    if len(clean_title) > 200:
        return False

    disallowed_phrases = (
        "fees earned",
        "shares of common stock",
        "beneficially owned",
        "page ",
        "proposal no.",
        "vote required",
        "percentile",
        "grant date",
        "award type",
        "value realized",
    )
    if any(phrase in clean_title.lower() for phrase in disallowed_phrases):
        return False

    return bool(re.search(r"[A-Za-z]", clean_title))


def _parse_board_members_from_text(soup):
    logger.info("Entering _parse_board_members_from_text")
    members = []
    seen = set()
    bio_keywords = ("director since", "former", "chief executive officer", "chair", "president", "ceo")

    for tag in soup.find_all(["p", "div"]):
        text = _normalize_board_member_text(tag.get_text(" ", strip=True))
        if len(text) < 20 or not any(keyword in text.lower() for keyword in bio_keywords):
            continue

        strong = tag.find(["b", "strong"])
        if not strong:
            continue

        name = _normalize_board_member_text(strong.get_text(" ", strip=True))
        remainder = text.replace(name, "", 1).strip(" ,:-")
        if not _is_person_like_name(name) or not _is_board_role_like(remainder):
            continue

        key = (name.lower(), remainder.lower())
        if key in seen:
            continue
        seen.add(key)
        members.append({"name": name, "title": remainder})

    return members


def _extract_board_members_from_def14a_html(html_text):
    logger.info("Entering _extract_board_members_from_def14a_html")
    if not html_text:
        return []

    soup = BeautifulSoup(html_text, "lxml")
    members = []
    seen = set()

    def _add_member(name, title):
        clean_name = _normalize_board_member_text(name)
        clean_title = _normalize_board_member_text(title)
        if not clean_name or not clean_title:
            return
        if clean_name.lower() == clean_title.lower():
            return
        if not _is_person_like_name(clean_name):
            return
        if not _is_board_role_like(clean_title):
            return
        key = (clean_name.lower(), clean_title.lower())
        if key in seen:
            return
        seen.add(key)
        members.append({"name": clean_name, "title": clean_title})

    for table in soup.find_all("table"):
        headers = _extract_table_headers(table)
        context_text = _get_table_context_text(table)
        if not _looks_like_director_table(headers, context_text):
            continue

        name_index = _find_column_index(headers, ("name", "director", "nominee"), default=0)
        title_index = _find_column_index(
            headers,
            ("occupation", "position", "title", "principal", "experience", "business"),
            default=1,
        )

        rows = table.find_all("tr")
        if headers and rows:
            rows = rows[1:]

        for row in rows:
            data_cells = row.find_all("td")
            if not data_cells:
                continue
            if max(name_index, title_index) >= len(data_cells):
                continue
            _add_member(
                data_cells[name_index].get_text(" ", strip=True),
                data_cells[title_index].get_text(" ", strip=True),
            )

    if members:
        return members

    return _parse_board_members_from_text(soup)


def collect_board_members_by_year(
    ticker,
    company_title,
    filings,
    data_dir=DATA_DIR,
):
    logger.info("Entering collect_board_members_by_year")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    board_members_by_year = []
    member_years = {}

    for filing in filings:
        year = filing.get("year")
        filing_url = filing.get("filing_url")
        if not year or not filing_url:
            continue

        filing_html = _fetch_proxy_filing_html(filing_url)
        if not filing_html:
            continue

        members = _extract_board_members_from_def14a_html(filing_html)
        board_members_by_year.append(
            {
                "year": int(year) if str(year).isdigit() else year,
                "members": members,
            }
        )

        for member in members:
            key = (member["name"], member["title"])
            member_years.setdefault(key, set()).add(int(year))

    board_members = [
        {
            "name": name,
            "title": title,
            "years_present": sorted(years),
        }
        for (name, title), years in member_years.items()
    ]
    board_members.sort(key=lambda member: (member["name"], member["title"]))

    output_path = data_dir / f"{ticker.lower()}_def14a_board_members.json"
    payload = {
        "ticker": ticker,
        "company_title": company_title,
        "form_type": "DEF 14A",
        "board_members": board_members,
        "board_members_by_year": board_members_by_year,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    record_document_lifecycle(
        file_path=output_path,
        stage="board_members_json_saved",
        ticker=ticker,
        form_type="DEF 14A",
    )
    return output_path

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

    folder, ticker, cik, filtered_filings = result
    docling_json_path = f"{folder}_docling.json"
    xbrl_json_path = f"{folder}.json"
    proxy_filings = _filter_def14a_filings(get_filings(cik))
    subsidiaries_json_path = collect_subsidiaries_by_year(
        cik=cik,
        ticker=ticker,
        company_title=company_title,
        form_type=form_type,
        filings=filtered_filings,
    )
    board_members_json_path = collect_board_members_by_year(
        ticker=ticker,
        company_title=company_title,
        filings=proxy_filings,
    )

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
        "subsidiaries_json_path": Path(subsidiaries_json_path).as_posix(),
        "board_members_json_path": Path(board_members_json_path).as_posix(),
    }


