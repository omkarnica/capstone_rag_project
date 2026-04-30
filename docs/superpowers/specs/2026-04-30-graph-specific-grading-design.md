# Graph-Specific Grading Design

Scope: prevent valid Neo4j graph retrieval rows from being discarded by the generic document grader, while leaving grading behavior for XBRL, Pinecone, filings, transcripts, patents, litigation, and other non-graph sources unchanged.

## Goal

Keep the current corrective-RAG grading stage for all sources, but introduce graph-specific grading behavior for `route == "graph"` so list-style graph answers are preserved when each retrieved row is individually relevant but atomic.

This should fix failures like:

- user asks for Apple board members in 2024
- Neo4j returns 8 correct rows
- generic per-row grading marks 7 rows as `no`
- downstream answer is built from only 1 surviving row

## Problem

The current document grader in [src/nodes/grader.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/nodes/grader.py) evaluates every retrieved document chunk one at a time with a generic prompt:

- `Reply "yes" if the chunk contains useful evidence for answering the question.`
- `Reply "no" if it is irrelevant, too broad, or too weak.`

That logic works better for larger chunks from filings, transcripts, and other corpus sources than it does for graph retrieval rows.

Graph retrieval often returns one entity or one relationship per row:

- one board member
- one subsidiary
- one filing
- one patent

For list-style graph questions, each row is intentionally small. A single row may look “too weak” in isolation even though the set of rows collectively answers the question. This causes the generic grader to over-prune graph results.

## Root Cause

The failure is specific to graph retrieval because of two combined issues:

1. graph rows are graded independently instead of as structured entity evidence
2. graph rows may use Cypher aliases such as `CompanyName`, `BoardMemberName`, `BoardMemberTitle`, and `YearsPresent`, which are not always fully normalized into the clearest possible evidence text before grading

The second issue is being addressed in [2026-04-30-graph-retrieval-evidence-formatting-design.md](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/docs/superpowers/specs/2026-04-30-graph-retrieval-evidence-formatting-design.md). This design covers the first issue: graph-specific grading behavior.

## Recommended Approach

Add a graph-only grading branch inside [grade_documents(...)](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/nodes/grader.py).

When `state["route"] == "graph"`:

- do not use the same “too weak” heuristic as generic chunk grading
- evaluate rows as graph evidence, not as standalone prose chunks
- preserve rows that directly match the company, year, and entity type requested by the query
- allow list-style graph answers to keep multiple relevant rows

All other routes keep the current grading logic unchanged.

## Non-Goals

This change should not:

- change grading behavior for `sql`
- change grading behavior for `filings`
- change grading behavior for `transcripts`
- change grading behavior for `patents`
- change grading behavior for `litigation`
- bypass grading globally
- remove the existing LLM grader for non-graph sources

## Desired Behavior

### Graph Queries

For `graph` route questions, grading should treat rows as atomic evidence units.

Examples:

- board-member rows for the requested company/year should remain relevant even if each row contains only one name
- subsidiary rows should remain relevant when they clearly identify a subsidiary of the requested company
- filing rows should remain relevant when they clearly identify the requested filing or section
- patent rows should remain relevant when they clearly identify the requested Apple patent or technology-domain relationship

### Other Sources

For all non-graph routes, grading should continue to use the current generic LLM prompt and filtering behavior.

## Implementation Shape

### 1. Route-Scoped Branch In `grade_documents`

In [src/nodes/grader.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/nodes/grader.py):

- inspect `state.get("route")`
- if route is not `graph`, keep current behavior
- if route is `graph`, call a dedicated graph-grading helper

### 2. Graph-Specific Relevance Helper

Add a helper dedicated to graph rows. It should be deterministic and scoped to graph evidence, rather than relying fully on the generic LLM chunk-grading prompt.

Recommended behavior:

- if the retrieved graph document came from `Knowledge Graph`, treat rows as relevant when they contain explicit entity evidence tied to the question
- preserve multiple rows for list-style graph queries
- use the question, rewritten question, and the row content together

The helper can still use a lightweight LLM prompt if necessary, but the preferred approach is deterministic graph-aware logic because:

- graph evidence is already structured
- list-style entity rows are easy to misgrade with generic prompts
- deterministic behavior is easier to reason about and test

## Graph-Specific Grading Rules

The graph-specific grader should consider rows relevant when they satisfy one or more of these conditions:

- the row explicitly names a graph entity requested by the query
- the row includes company and year evidence aligned with the query
- the row is part of a graph list answer and matches the requested entity family

Examples:

- `Apple board member in 2024: Alex Gorsky ...` should be relevant for board-member queries about Apple in 2024
- `Apple subsidiary: Apple Operations International Limited` should be relevant for subsidiary queries
- `Apple filing: 2024 10-K ...` should be relevant for filing queries
- `Apple patent: US1234567 ...` should be relevant for patent queries

The graph-specific grader should not reject these rows merely because each row covers only one item in a longer answer.

## List-Style Query Support

The graph-specific branch should explicitly support list-style prompts such as:

- `who are`
- `list`
- `what are the board members`
- `what subsidiaries`
- `which patents`

For these queries, retaining multiple graph rows is expected behavior, not evidence of weak retrieval.

## Error Handling

- If graph-specific grading cannot confidently classify a row, default to keeping the row rather than discarding likely-valid graph evidence.
- If graph-specific logic encounters an unexpected graph document shape, it should fail open to preserve graph results instead of over-pruning them.
- If the route is not `graph`, none of this fallback behavior should apply.

## Testing

Add tests proving:

- graph route preserves all relevant board-member rows in a list-style query
- graph route preserves all relevant subsidiary rows in a list-style query
- graph route keeps relevant filing and patent evidence rows
- non-graph routes still use existing grading behavior
- graph-specific grading does not change `sql`, `filings`, `transcripts`, `patents`, or `litigation` route logic

## Out of Scope

- changing router behavior
- changing planner behavior
- changing the Neo4j schema
- changing non-graph grading prompts
- changing answer generation outside what follows naturally from receiving the full graph evidence set
