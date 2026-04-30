# Graph Retrieval Evidence Formatting Design

Scope: improve Neo4j retrieval output so graph-based answers are grounded in explicit natural-language evidence for board members, subsidiaries, filings, filing sections, patents, and technology domains.

## Goal

Keep the existing graph retrieval flow:

1. user question
2. graph route
3. LLM-generated read-only Cypher
4. Neo4j query execution
5. retrieved documents passed into answer generation

But change step 5 so graph results are formatted as self-explanatory evidence rather than generic key-value rows. This should improve answer quality for any graph retrieval, not just board-member queries.

## Problem

Today [src/graph_retrieval.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/graph_retrieval.py) serializes each Neo4j row into generic lines such as:

- `BoardMemberName: Art Levinson`
- `BoardMemberTitle: Founder and CEO, Calico`

This preserves raw data, but it leaves too much semantic work for the answer generator. The generator in [src/nodes/generator.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/nodes/generator.py) is instructed to answer only from explicit evidence. If the row does not clearly say that Art Levinson is an Apple board member in 2024, the model may answer cautiously even when the graph query is basically correct.

The issue is therefore not routing. The issue is that graph retrieval outputs are too weakly formatted for downstream answer generation.

## Recommended Approach

Use a hybrid approach:

1. strengthen the graph Cypher prompt so the LLM prefers richer semantic columns
2. transform Neo4j rows into explicit natural-language evidence lines before they reach the answer generator

This is better than a prompt-only change because it does not rely entirely on the graph LLM returning ideal aliases every time. It is also better than formatting alone because stronger Cypher returns improve the fidelity and consistency of the formatter.

## Existing Behavior

In [src/graph_retrieval.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/graph_retrieval.py):

- `_graph_prompt(...)` asks the graph LLM to produce read-only Cypher
- `generate_cypher(...)` runs the graph LLM and validates the result
- `retrieve_graph_docs(...)` executes the Cypher in Neo4j
- `_row_to_doc(...)` converts each row into a generic document by rendering `key: value` pairs

This final conversion is where semantic clarity is currently lost.

## Desired Behavior

### General Rule

Every graph-retrieved document should make the underlying fact explicit in plain language, while still retaining enough metadata for debugging and traceability.

The answer generator should be able to read the document content and understand:

- what kind of entity this is
- how it relates to the company
- what year or time range applies, if any
- which fields are the core evidence

### Coverage

This behavior should apply to graph retrieval across at least these result families:

- board members
- subsidiaries
- filings
- filing sections
- patents
- technology domains

Unknown row shapes should still fall back to generic key-value formatting instead of failing.

## Evidence Formatting Strategy

### Board Members

Board-member rows should format into direct evidence such as:

- `Apple board member in 2024: Art Levinson | Title: Founder and CEO, Calico`
- `Apple board member: Tim Cook | Title: CEO, Apple | Years present: 2024, 2025`

The formatter should use fields like:

- member name
- member title
- company ticker or company name
- years present
- current status if available

If the row contains a year filter result, that should be reflected in the sentence.

### Subsidiaries

Subsidiary rows should format into evidence such as:

- `Apple subsidiary: Apple Operations International Limited`
- `Apple subsidiary in 2024: Apple Sales International | Source form: 10-K`

The formatter should use fields like:

- subsidiary name
- company ticker or company name
- year
- source form type

### Filings

Filing rows should format into evidence such as:

- `Apple filing: 2024 10-K | Filing ID: AAPL_10-K_2024`
- `Apple filing: 2024 10-Q | Source file: aapl_10-q_docling.json`

The formatter should use fields like:

- ticker or company
- year
- form type
- filing id
- source file

### Filing Sections

Section rows should format into evidence such as:

- `Apple 2024 10-K section: Risk Factors | Section ID: AAPL_10-K_2024_section_12`
- `Apple filing section: Foreign Currency Risk | Year: 2024 | Form: 10-K`

If section text is present, it should be included after the evidence header rather than hidden inside raw keys.

### Patents

Patent rows should format into evidence such as:

- `Apple patent: US1234567 | Title: Wireless security system | Grant date: 2024-03-15`
- `Apple patent in domain G06: ...`

The formatter should use fields like:

- patent id
- patent title
- grant date
- grant year
- assignee or ticker
- domain if present

### Technology Domains

Technology-domain rows should format into evidence such as:

- `Apple technology domain: G06 | Computing & Data Processing`
- `Technology domain linked to Apple patent: H04 | Telecommunications & Signal Processing`

## Cypher Prompt Changes

The graph prompt should continue to enforce read-only Cypher, but should add instructions that make returned rows easier to format and answer from:

- return semantic aliases instead of opaque abbreviations where possible
- include company and year fields when available
- for board-member questions, prefer `years_present`, `is_current`, and explicit year filtering
- for subsidiary questions, include company and source-form context when available
- for filing questions, include form type, year, filing id, and source file
- for patent questions, include patent id, title, grant date, and domain fields

This does not require hard-coding Cypher templates for each query type. The graph LLM can still generate Cypher dynamically, but the prompt should bias the result toward answer-friendly columns.

## Implementation Shape

In [src/graph_retrieval.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/graph_retrieval.py):

1. keep `generate_cypher(...)` and `retrieve_graph_docs(...)` as the query-time entry points
2. replace the current single generic `_row_to_doc(...)` behavior with:
   - row-shape detection
   - specialized evidence formatting helpers
   - generic fallback formatting for unmatched row shapes
3. preserve metadata such as the executed Cypher in the returned document metadata

This keeps the change isolated to graph retrieval and avoids touching router, planner, or generator logic.

## Error Handling

- If graph rows do not match a known pattern, fall back to generic key-value formatting.
- Do not fail retrieval because a row is unfamiliar.
- Preserve the Cypher string in metadata for debugging even when content is transformed into natural-language evidence.

## Testing

Add tests for:

- board-member row formatting
- subsidiary row formatting
- filing row formatting
- filing-section row formatting
- patent row formatting
- technology-domain row formatting
- generic fallback for unknown row shapes
- `retrieve_graph_docs(...)` returning semantic graph evidence docs instead of only raw key-value output

## Out of Scope

- changing the Neo4j schema
- changing router or planner behavior
- replacing LLM-generated Cypher with hard-coded query templates
- redesigning the generator prompt outside what is necessary to consume clearer graph evidence
