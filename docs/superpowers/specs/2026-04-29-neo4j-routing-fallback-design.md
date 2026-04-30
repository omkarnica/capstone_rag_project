# Neo4j Routing Fallback Design

Scope: deterministic routing to Neo4j for board-member and subsidiary questions, plus a fallback from filings retrieval to the knowledge graph when Pinecone retrieval fails or returns no useful evidence.

## Goal

Preserve the current filings-first retrieval path for narrative SEC content, while guaranteeing that:

- board-member and subsidiary questions route directly to the Neo4j knowledge graph
- filing-oriented questions fall back to Neo4j when filings retrieval throws an exception
- filing-oriented questions fall back to Neo4j when filings retrieval returns no relevant evidence after grading

## Existing Behavior

- [src/nodes/router.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/nodes/router.py) delegates routing to an LLM with only prompt-level guidance for `graph`.
- [src/nodes/planner.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/nodes/planner.py) can hint `filings`, `sql`, `transcripts`, `patents`, `litigation`, `contradiction`, or `llm_direct`, but does not explicitly detect board-member or subsidiary prompts as graph questions.
- [src/nodes/retriever.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/nodes/retriever.py) dispatches to Pinecone-backed filings retrieval when route is `filings`, and to [src/graph_retrieval.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/graph_retrieval.py) only when route is already `graph`.
- [src/graph.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/graph.py) currently rewrites or falls back to web search when relevant filings docs are unavailable; it does not attempt Neo4j as a secondary evidence source.

## Required Behavior

### Deterministic Graph Routing

Questions about these topics should bypass the LLM router and immediately route to `graph`:

- board members
- directors
- board composition
- subsidiaries
- corporate hierarchy
- ownership structure
- entity relationship questions clearly covered by the Neo4j schema

This should be implemented as a lightweight keyword and phrase matcher before the LLM router executes.

### Filings-to-Graph Fallback

If a question is initially routed to `filings`, the pipeline should attempt Neo4j once before rewrite or web fallback when either of these happens:

- filings retrieval raises an exception
- filings retrieval returns zero documents
- filings retrieval returns documents but grading reduces them to zero relevant documents

The fallback must happen at most once per query to avoid loops.

### Query-Time Knowledge Graph Retrieval

Knowledge graph retrieval remains centralized in [src/graph_retrieval.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/graph_retrieval.py):

1. accept the user question
2. use the graph LLM to turn the question into read-only Cypher
3. execute the Cypher against Neo4j
4. format Neo4j rows as retrieved documents for downstream grading and generation

No Cypher generation logic should be duplicated in router or retriever code.

## Configuration

Neo4j config should prefer environment variables over YAML fallback. The retrieval and ingestion code should read:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`

## Proposed Code Changes

### 1. Graph Topic Detection

Add a shared helper in the routing layer that normalizes the question and checks for graph-oriented terms. The helper should favor phrase-level matches such as:

- `board members`
- `board member`
- `board of directors`
- `directors`
- `subsidiaries`
- `subsidiary`
- `corporate hierarchy`
- `ownership structure`

Use that helper in both router and planner so they stay aligned.

### 2. Router Update

In [src/nodes/router.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/nodes/router.py):

- run deterministic graph-topic detection before calling the router LLM
- immediately return `route="graph"` when the query matches a graph topic
- retain the current LLM routing behavior for all other questions

### 3. Planner Update

In [src/nodes/planner.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/nodes/planner.py):

- add graph-specific markers
- update `_guess_route_hint()` so graph-topic questions return `graph`
- keep the rest of the planning and decomposition logic unchanged

### 4. Retriever State Update

In [src/nodes/retriever.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/nodes/retriever.py):

- record whether filings retrieval failed with an exception
- record whether filings retrieval returned zero docs
- preserve the current graph route dispatch to `retrieve_graph_docs(...)`

The retriever should annotate state so the graph orchestration layer can decide whether to fall back.

### 5. Graph Orchestration Update

In [src/graph.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/graph.py):

- add a conditional branch after doc grading
- if the current route is `filings` and there are zero relevant docs, try `graph` once before rewrite or web fallback
- if filings retrieval failed before grading, try `graph` once before rewrite or web fallback
- mark state so this fallback cannot recur

### 6. Neo4j Environment Loading

In [src/graph_retrieval.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/graph_retrieval.py) and [src/Knowledge graph/kg.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/Knowledge%20graph/kg.py):

- load `.env` values before resolving Neo4j settings
- preserve YAML defaults as a final fallback

## Error Handling

- If filings retrieval fails, do not stop the query immediately; attempt Neo4j fallback first.
- If Neo4j retrieval also fails, continue with the existing rewrite or web fallback path.
- If graph-topic routing sends a question to Neo4j and Neo4j fails, keep existing downstream resilience behavior rather than crashing the whole pipeline.

## Testing

Add tests covering:

- router sends board-member queries to `graph`
- router sends subsidiary queries to `graph`
- planner hints `graph` for those same topics
- filings retrieval exception triggers graph fallback
- filings retrieval with zero docs triggers graph fallback
- filings retrieval with zero relevant docs after grading triggers graph fallback
- graph fallback happens only once
- Neo4j settings resolve correctly from `.env`

## Out of Scope

- changing the Neo4j schema
- modifying Cypher prompt semantics beyond what current graph retrieval already uses
- expanding the graph schema beyond current board-member, subsidiary, filing, section, patent, and domain coverage
