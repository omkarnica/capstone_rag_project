"""
Contradiction detection for M&A Oracle due diligence.

detect_contradiction:
    Compares what management said on earnings calls (transcripts → Pinecone)
    against what the financial filings actually report (XBRL → PostgreSQL SQL).
    Sends both to Gemini and returns a structured contradiction assessment.

run_due_diligence:
    Runs detect_contradiction across a predefined set of key financial metrics
    for a company and fiscal year. Returns a list of findings sorted by
    contradiction score descending.

Contradiction score: 0–10 (0 = fully consistent, 10 = direct contradiction).
Severity mapping: 0–2 = low, 3–5 = medium, 6–8 = high, 9–10 = critical.

LLM: gemini-2.5-flash via google-genai SDK + GCP Vertex AI
Sources: src/nl_sql/pipeline.ask(), src/transcripts/retrieval.generate_transcript_answer()
"""

from __future__ import annotations

import time

from google import genai
from google.genai import types

from src.nl_sql.pipeline import ask
from src.transcripts.retrieval import generate_transcript_answer
from src.utils.logger import get_logger

logger = get_logger(__name__)

_GCP_PROJECT  = "codelab-2-485215"
_GCP_LOCATION = "us-central1"
_LLM_MODEL    = "gemini-2.5-flash"

_SYSTEM_PROMPT = """\
You are a forensic financial analyst performing M&A due diligence.
You are given two pieces of information about the same company metric and period:
  1. STRUCTURED DATA: the official XBRL-reported figure from SEC financial filings.
  2. MANAGEMENT STATEMENT: what executives said on the earnings call.

Your job is to identify whether management's statements are consistent with the reported figures.

Respond ONLY as valid JSON with exactly these fields:
{
  "contradiction_score": <integer 0-10>,
  "contradiction_detected": <true|false>,
  "severity": <"low"|"medium"|"high"|"critical">,
  "xbrl_summary": "<one sentence summarizing the XBRL figure>",
  "transcript_summary": "<one sentence summarizing management's claim>",
  "explanation": "<2-3 sentences explaining the comparison and any discrepancy>"
}

Scoring guide:
  0-2  = Fully consistent or management slightly overstates a positive (normal)
  3-5  = Noticeable discrepancy — management tone doesn't match data magnitude
  6-8  = Material contradiction — management claims contradict the reported figure
  9-10 = Direct contradiction — management explicitly states the opposite of what filings show

Do not include any text outside the JSON object."""

# Predefined metric templates for due diligence sweeps
# Each entry: (metric_label, xbrl_question_template, transcript_question_template)
_DUE_DILIGENCE_METRICS: list[tuple[str, str, str]] = [
    (
        "Total Revenue",
        "What is {company}'s total net sales or revenue for {period}?",
        "What did {company} report about total revenue or net sales in {period}?",
    ),
    (
        "Net Income",
        "What is {company}'s net income for {period}?",
        "What did {company} say about net income or profitability in {period}?",
    ),
    (
        "Gross Margin",
        "What is {company}'s gross margin or gross profit for {period}?",
        "What did {company} say about gross margin in {period}?",
    ),
    (
        "Operating Income",
        "What is {company}'s operating income for {period}?",
        "What did {company} say about operating performance in {period}?",
    ),
    (
        "R&D Spending",
        "What is {company}'s research and development expense for {period}?",
        "What did {company} say about research and development investment in {period}?",
    ),
    (
        "Cash and Equivalents",
        "What is {company}'s cash and cash equivalents for {period}?",
        "What did {company} say about cash position or liquidity in {period}?",
    ),
]

_genai_client: genai.Client | None = None


def _get_genai() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(
            vertexai=True,
            project=_GCP_PROJECT,
            location=_GCP_LOCATION,
        )
    return _genai_client


def _severity_from_score(score: int) -> str:
    if score <= 2:
        return "low"
    if score <= 5:
        return "medium"
    if score <= 8:
        return "high"
    return "critical"


def _parse_gemini_json(raw: str) -> dict:
    """Extract and parse the JSON object from Gemini's response."""
    import json
    import re
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: return a minimal structure so the pipeline doesn't crash
        logger.warning("Failed to parse Gemini JSON response", extra={"raw": raw[:500]})
        return {
            "contradiction_score":    0,
            "contradiction_detected": False,
            "severity":               "low",
            "xbrl_summary":           "Parse error",
            "transcript_summary":     "Parse error",
            "explanation":            raw[:500],
        }


def detect_contradiction(
    company: str,
    metric_label: str,
    xbrl_question: str,
    transcript_question: str,
    transcript_company: str,
    period_label: str,
    transcript_period_start: str,
    transcript_period_end: str,
) -> dict:
    """
    Compare a single financial metric between XBRL data and earnings call statements.

    Args:
        company:                  Company name for display (e.g. "Apple Inc.").
        metric_label:             Human-readable metric name (e.g. "Total Revenue").
        xbrl_question:            NL question for the NL-to-SQL pipeline.
        transcript_question:      NL question for the transcript retrieval pipeline.
        transcript_company:       Exact company_title value in Pinecone transcripts.
        period_label:             Human-readable period (e.g. "fiscal Q4 2024").
        transcript_period_start:  ISO date for transcript date filter lower bound.
        transcript_period_end:    ISO date for transcript date filter upper bound.

    Returns:
        Dict with keys: company, metric, period, xbrl_result, transcript_result,
        contradiction_score (0-10), contradiction_detected (bool), severity,
        xbrl_summary, transcript_summary, explanation.
    """
    logger.info(
        "Contradiction check start",
        extra={"company": company, "metric": metric_label, "period": period_label},
    )

    # Step 1: Get ground truth from XBRL (structured SQL)
    xbrl_answer = "No data available."
    xbrl_sql = ""
    try:
        xbrl_result = ask(xbrl_question)
        xbrl_answer = xbrl_result.get("answer", "No data available.")
        xbrl_sql    = xbrl_result.get("sql", "")
    except Exception as exc:
        logger.warning("XBRL query failed", extra={"error": str(exc), "metric": metric_label})
        xbrl_answer = f"XBRL query error: {exc}"

    # Step 2: Get management's narrative from earnings transcripts
    transcript_answer = "No transcript data available."
    try:
        transcript_result = generate_transcript_answer(
            query=transcript_question,
            company=transcript_company,
            period_start=transcript_period_start,
            period_end=transcript_period_end,
        )
        transcript_answer = transcript_result.get("answer", "No transcript data available.")
    except Exception as exc:
        logger.warning("Transcript query failed", extra={"error": str(exc), "metric": metric_label})
        transcript_answer = f"Transcript query error: {exc}"

    # Step 3: Ask Gemini to compare the two
    # Brief pause so rapid sequential calls don't exhaust the Vertex AI quota
    time.sleep(2)

    user_prompt = (
        f"Company: {company}\n"
        f"Metric: {metric_label}\n"
        f"Period: {period_label}\n\n"
        f"STRUCTURED DATA (XBRL from SEC filing):\n{xbrl_answer}\n\n"
        f"MANAGEMENT STATEMENT (earnings call transcript):\n{transcript_answer}"
    )

    parsed: dict = {}
    try:
        client = _get_genai()
        response = client.models.generate_content(
            model=_LLM_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=_SYSTEM_PROMPT),
        )
        parsed = _parse_gemini_json(response.text or "")
    except Exception as exc:
        logger.warning(
            "Contradiction comparison LLM call failed",
            extra={"error": str(exc), "metric": metric_label},
        )
        parsed = {
            "contradiction_score":    0,
            "contradiction_detected": False,
            "severity":               "low",
            "xbrl_summary":           xbrl_answer[:200],
            "transcript_summary":     transcript_answer[:200],
            "explanation":            f"Comparison unavailable due to API error: {exc}",
        }

    score    = int(parsed.get("contradiction_score", 0))
    severity = parsed.get("severity") or _severity_from_score(score)

    result = {
        "company":                company,
        "metric":                 metric_label,
        "period":                 period_label,
        "xbrl_answer":            xbrl_answer,
        "xbrl_sql":               xbrl_sql,
        "transcript_answer":      transcript_answer,
        "contradiction_score":    score,
        "contradiction_detected": bool(parsed.get("contradiction_detected", score >= 3)),
        "severity":               severity,
        "xbrl_summary":           parsed.get("xbrl_summary", ""),
        "transcript_summary":     parsed.get("transcript_summary", ""),
        "explanation":            parsed.get("explanation", ""),
    }

    logger.info(
        "Contradiction check complete",
        extra={
            "company":  company,
            "metric":   metric_label,
            "score":    score,
            "severity": severity,
        },
    )
    return result


def run_due_diligence(
    company: str,
    transcript_company: str,
    fiscal_year: int,
    quarter: str = "Q4",
) -> list[dict]:
    """
    Run contradiction detection across all key financial metrics for a company.

    Covers: Total Revenue, Net Income, Gross Margin, Operating Income,
    R&D Spending, Cash and Equivalents.

    Args:
        company:             Company name for XBRL queries (e.g. "Apple Inc.").
        transcript_company:  Exact company_title in Pinecone (e.g. "Apple Inc.").
        fiscal_year:         Fiscal year to analyze (e.g. 2024).
        quarter:             Fiscal quarter label (e.g. "Q4", "FY").

    Returns:
        List of contradiction result dicts, sorted by contradiction_score descending.
    """
    # Map quarter to approximate date ranges for transcript filter
    _quarter_dates: dict[str, tuple[str, str]] = {
        "Q1": (f"{fiscal_year}-01-01", f"{fiscal_year}-03-31"),
        "Q2": (f"{fiscal_year}-04-01", f"{fiscal_year}-06-30"),
        "Q3": (f"{fiscal_year}-07-01", f"{fiscal_year}-09-30"),
        "Q4": (f"{fiscal_year}-07-01", f"{fiscal_year + 1}-01-31"),
        "FY": (f"{fiscal_year}-01-01", f"{fiscal_year + 1}-03-31"),
    }
    period_start, period_end = _quarter_dates.get(quarter, (f"{fiscal_year}-01-01", f"{fiscal_year + 1}-03-31"))
    period_label = f"fiscal {quarter} {fiscal_year}"

    findings: list[dict] = []

    for metric_label, xbrl_tmpl, transcript_tmpl in _DUE_DILIGENCE_METRICS:
        xbrl_q       = xbrl_tmpl.format(company=company, period=period_label)
        transcript_q = transcript_tmpl.format(company=company, period=period_label)

        finding = detect_contradiction(
            company=company,
            metric_label=metric_label,
            xbrl_question=xbrl_q,
            transcript_question=transcript_q,
            transcript_company=transcript_company,
            period_label=period_label,
            transcript_period_start=period_start,
            transcript_period_end=period_end,
        )
        findings.append(finding)
        # Pause between metrics to stay within Vertex AI rate limits
        time.sleep(5)

    findings.sort(key=lambda f: f["contradiction_score"], reverse=True)

    high_or_above = [f for f in findings if f["contradiction_score"] >= 6]
    logger.info(
        "Due diligence sweep complete",
        extra={
            "company":      company,
            "period":       period_label,
            "total_checks": len(findings),
            "high_or_above": len(high_or_above),
        },
    )
    return findings


if __name__ == "__main__":
    findings = run_due_diligence(
        company="MICROSOFT CORP",
        transcript_company="MICROSOFT CORP",
        fiscal_year=2024,
        quarter="FY",
    )

    print(f"\n{'='*60}")
    print(f"DUE DILIGENCE REPORT — Microsoft Corp FY2024")
    print(f"{'='*60}")
    for f in findings:
        flag = "🔴" if f["contradiction_score"] >= 6 else ("🟡" if f["contradiction_score"] >= 3 else "🟢")
        print(f"\n{flag} [{f['severity'].upper()}] {f['metric']} — Score: {f['contradiction_score']}/10")
        print(f"  XBRL:       {f['xbrl_summary']}")
        print(f"  Transcript: {f['transcript_summary']}")
        print(f"  {f['explanation']}")
