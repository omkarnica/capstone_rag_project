"""
Natural language to SQL pipeline for M&A Oracle.

ask(question) takes a plain-English question, sends it to Gemini via
the google-genai SDK + GCP Vertex AI, extracts the generated SQL, executes
it against PostgreSQL, and returns a structured result dict.

On SQL execution failure the pipeline retries once — it appends the error
message to the prompt so Gemini can self-correct the query.

LLM: gemini-2.5-flash via google-genai SDK (no API keys — GCP creds only)
DB:  PostgreSQL, get_connection() from xbrl/loader.py
"""

from __future__ import annotations

from google import genai
from google.genai import types

from src.utils.exceptions import DatabaseError, db_error_boundary
from src.utils.logger import get_logger
from src.xbrl.loader import get_connection

logger = get_logger(__name__)

_GCP_PROJECT  = "codelab-2-485215"
_GCP_LOCATION = "us-central1"
_MODEL        = "gemini-2.5-flash"

_client = genai.Client(
    vertexai=True,
    project=_GCP_PROJECT,
    location=_GCP_LOCATION,
)

_SCHEMA = """
Table: filings
Columns: adsh (TEXT, primary key), cik (INTEGER), name (TEXT),
         sic (INTEGER), form (TEXT), period (DATE),
         fiscal_year (INTEGER), fiscal_period (TEXT), filed (DATE)
Notes: name is always UPPERCASE e.g. 'MICROSOFT CORP'.
       form values include '10-K', '10-Q'.
       fiscal_period values include 'FY', 'Q1', 'Q2', 'Q3', 'Q4'.
       adsh is the join key to facts table.

Table: facts
Columns: id (SERIAL), adsh (TEXT, FK to filings.adsh), tag (TEXT),
         version (TEXT), ddate (DATE), qtrs (INTEGER),
         uom (TEXT), value (NUMERIC)
Notes: tag contains raw XBRL tags e.g. 'Revenue', 'NetIncome',
       'TotalAssets', 'OperatingIncome', 'StockholdersEquity'.
       qtrs=4 means annual figure. qtrs=1 means quarterly figure.
       Always join facts to filings on adsh to get company name.
       Always filter uom='USD' for financial values.

Example tags: Revenue, NetIncome, TotalAssets, OperatingIncome,
              StockholdersEquity, AssetsCurrent, CashAndCashEquivalentsAtCarryingValue

Table: patents
Columns: patent_id (TEXT, primary key), patent_title (TEXT),
         assignee_organization (TEXT), grant_date (DATE, format YYYY-MM-DD),
         cpc_codes (TEXT[], a PostgreSQL array), citation_count (INTEGER)
Notes: Use for structured questions about patent counts, date ranges,
       CPC code filtering, and citation ranking.
       For CPC filtering use = ANY(cpc_codes) or && ARRAY['code']::text[], not LIKE.
       Do NOT join patents to filings or facts — it is an independent table.

Table: transcripts
Columns: accession_no (TEXT, primary key), company_name (TEXT),
         filed_date (DATE), period_of_report (DATE), form_type (TEXT)
Notes: company_name is stored exactly as e.g. 'MICROSOFT CORP' or 'Apple Inc.' — use ILIKE for matching.
       form_type values include '8-K'.
       Use for structured questions that filter by company or date range.
       Do NOT use for questions about transcript content — those require vector search, not SQL.
       Do NOT join transcripts to filings or facts — it is an independent table.
"""

_SYSTEM_PROMPT = """You are a SQL expert. Generate a single valid PostgreSQL query to answer \
the user's question using only the tables and columns described above.
Rules:
- Return ONLY the SQL query, no explanation, no markdown, no backticks
- Always use ILIKE for company name matching
- Always include LIMIT 20 unless the question asks for aggregates
- For revenue/income questions, filter qtrs=4 for annual, qtrs=1 for quarterly
- Always filter uom='USD' for monetary values
- Use canonical tag names exactly as listed above
- When tag is ambiguous, prefer the shorter canonical form (e.g. 'Revenue' not 'Revenues')
- For total company revenue or income questions, use MAX(value) when multiple segment values exist"""


def _call_llm(messages: list[dict]) -> str:
    """
    Call Gemini via the google-genai SDK with Vertex AI credentials.

    Extracts the system message (if any) and user message from the
    OpenAI-style message list and maps them to the genai SDK parameters.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.

    Returns:
        Raw text content of the model response.
    """
    system_content = next(
        (m["content"] for m in messages if m["role"] == "system"), None
    )
    user_content = next(
        (m["content"] for m in messages if m["role"] == "user"), ""
    )

    config = types.GenerateContentConfig(
        system_instruction=system_content,
    ) if system_content else None

    response = _client.models.generate_content(
        model=_MODEL,
        contents=user_content,
        config=config,
    )
    return response.text or ""


def _extract_sql(raw: str) -> str:
    """
    Strip accidental markdown fences, language tags, and whitespace.

    Gemini occasionally wraps output in ```sql ... ``` despite instructions.
    This strips all of that and returns bare SQL.
    """
    text = raw.strip()
    # Remove opening fence (```sql, ```postgresql, ```)
    if text.startswith("```"):
        text = text.lstrip("`")
        # Remove optional language identifier on the first line
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            if not first_line.strip().upper().startswith("SELECT"):
                text = rest
    # Remove closing fence
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    return text.strip()


def _build_messages(question: str, extra_context: str = "") -> list[dict]:
    """
    Build the message list for a given question.

    Args:
        question:      The user's natural-language question.
        extra_context: Optional suffix appended to the user message
                       (used on retry to include the previous SQL error).
    """
    user_content = (
        f"Schema:\n{_SCHEMA}\n\nQuestion: {question}"
        + (f"\n\n{extra_context}" if extra_context else "")
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def _format_rows(columns: list[str], rows: list[tuple], max_rows: int = 20) -> str:
    """
    Format query results as a plain-text table for the synthesis prompt.

    Args:
        columns: Column name strings.
        rows:    List of row tuples from psycopg2.
        max_rows: Maximum rows to include (capped at 20).

    Returns:
        A header line + one line per row, tab-separated.
    """
    lines = ["\t".join(columns)]
    for row in rows[:max_rows]:
        lines.append("\t".join(str(v) for v in row))
    return "\n".join(lines)


def _synthesize_answer(question: str, sql: str, columns: list[str], rows: list[tuple]) -> str:
    """
    Call Gemini a second time to produce a natural-language answer from query results.

    Args:
        question: The original user question.
        sql:      The SQL that was executed.
        columns:  Column names from the result set.
        rows:     Row tuples returned by PostgreSQL (capped at 20 in the prompt).

    Returns:
        A 1-2 sentence plain-English answer.
    """
    formatted = _format_rows(columns, rows)
    prompt = (
        f"Given this question: {question}\n"
        f"This SQL was executed: {sql}\n"
        f"These are the results:\n{formatted}\n\n"
        "Provide a clear, concise answer to the question in 1-2 sentences.\n"
        "Use plain numbers with commas. If multiple rows exist, summarize them.\n"
        "If the result is a single financial figure, state it directly."
    )
    response = _client.models.generate_content(model=_MODEL, contents=prompt)
    answer = (response.text or "").strip()
    logger.info("Synthesized answer", extra={"question": question, "answer": answer})
    return answer


def _execute_sql(sql: str) -> tuple[list[tuple], list[str]]:
    """
    Execute a SQL query and return (rows, column_names).

    Args:
        sql: A validated SELECT statement.

    Returns:
        (rows, columns) — rows is a list of tuples, columns is a list of
        column name strings.

    Raises:
        DatabaseError: wraps any psycopg2.Error.
    """
    conn = get_connection()
    try:
        with db_error_boundary("SELECT", table="nl_sql"):
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
        return rows, columns
    finally:
        conn.close()


def ask(question: str) -> dict:
    """
    Translate a natural-language question into SQL, execute it, and return results.

    Flow:
    1. Build system + user messages with the embedded schema.
    2. Call Gemini via LiteLLM + GCP credentials.
    3. Strip markdown fences from the response.
    4. Validate: raise ValueError if the result does not start with SELECT.
    5. Execute against PostgreSQL via get_connection().
    6. On DB error: retry once with the error appended to the prompt.

    Args:
        question: A plain-English question about the M&A Oracle database,
                  e.g. "What is Microsoft's revenue for fiscal year 2024?"

    Returns:
        dict with keys:
            question (str)  — the original question
            sql      (str)  — the final executed SQL
            results  (list) — list of row tuples
            columns  (list) — list of column name strings

    Raises:
        ValueError:     if the LLM response does not start with SELECT
                        (after stripping) on both the initial attempt and retry.
        DatabaseError:  if the SQL fails on the retry attempt as well.
    """
    # ── First attempt ─────────────────────────────────────────────────────────
    messages = _build_messages(question)
    raw = _call_llm(messages)
    sql = _extract_sql(raw)

    if not sql.upper().startswith("SELECT"):
        logger.warning(
            "LLM returned non-SELECT response",
            extra={"question": question, "raw_response": raw},
        )
        raise ValueError(f"LLM did not return a SELECT statement. Raw response:\n{raw}")

    logger.info("Generated SQL", extra={"question": question, "sql": sql})

    try:
        rows, columns = _execute_sql(sql)
        logger.info(
            "SQL executed successfully",
            extra={"row_count": len(rows), "columns": columns},
        )
        answer = _synthesize_answer(question, sql, columns, rows)
        return {
            "question": question,
            "sql":      sql,
            "results":  rows,
            "columns":  columns,
            "answer":   answer,
        }

    except DatabaseError as first_error:
        # ── Retry with error feedback ──────────────────────────────────────
        logger.warning(
            "SQL execution failed — retrying with error context",
            extra={"sql": sql, "error": str(first_error)},
        )

        retry_context = (
            f"The previous SQL query failed with this error:\n{first_error}\n\n"
            f"Previous SQL:\n{sql}\n\n"
            "Please generate a corrected SQL query."
        )
        retry_messages = _build_messages(question, extra_context=retry_context)
        retry_raw = _call_llm(retry_messages)
        retry_sql = _extract_sql(retry_raw)

        if not retry_sql.upper().startswith("SELECT"):
            raise ValueError(
                f"LLM did not return a SELECT statement on retry. Raw response:\n{retry_raw}"
            )

        logger.info(
            "Retried SQL generated",
            extra={"question": question, "sql": retry_sql},
        )

        rows, columns = _execute_sql(retry_sql)
        logger.info(
            "Retry SQL executed successfully",
            extra={"row_count": len(rows), "columns": columns},
        )
        answer = _synthesize_answer(question, retry_sql, columns, rows)
        return {
            "question": question,
            "sql":      retry_sql,
            "results":  rows,
            "columns":  columns,
            "answer":   answer,
        }
