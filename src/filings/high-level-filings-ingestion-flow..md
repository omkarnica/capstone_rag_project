run_filings_pipeline(company_title, form_type, namespace)
  |
  v
ingestion_filing()
  |
  |-- find company CIK from SEC company_tickers.json
  |-- fetch filing manifest from SEC submissions API
  |-- download filing HTML files
  |-- extract XBRL JSON
  `-- convert cleaned filing HTML to Docling JSON
  |
  v
docling_json_to_pinecone_chunks()
  |
  |-- extract text nodes
  |-- clean noisy SEC/filing text
  |-- build semantic blocks
  |-- create text chunks
  `-- add structured table chunks
  |
  v
save_chunks_to_json()
  |
  v
run_raptor_pipeline()
  |
  |-- load chunks JSON
  |-- build RAPTOR leaf + summary tree
  |-- summarize clusters with Gemini
  |-- embed nodes with Pinecone inference
  |-- upsert vectors to Pinecone
  `-- save local RAPTOR tree map
  |
  v
verify_raptor_tree()

raptor_retrieval.py is not part of ingestion. It is the query-time retrieval layer after vectors already exist in Pinecone:
Question from user
  |
  v
raptor_retrieve()
  |
  v
embed query
  |
  v
query Pinecone
  |
  v
expand summary matches to child chunks
  |
  v
rerank/build context for LLM

