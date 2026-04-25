# BigQuery Audit Log — Admin Setup Guide

One-time setup for the `codelab-2-485215` project. Only the project admin needs to
do this. Once done, the application writes audit records automatically on every query.

---

## Step 1 — Enable the BigQuery API

In **GCP Console → APIs & Services → Enable APIs and Services**, search for
**BigQuery API** and enable it (it may already be enabled).

Or via CLI:
```
gcloud services enable bigquery.googleapis.com --project=codelab-2-485215
```

---

## Step 2 — Create the dataset

In **BigQuery Studio → Explorer → codelab-2-485215 → Create dataset**:

| Field | Value |
|---|---|
| Dataset ID | `ma_oracle` |
| Location type | Region |
| Region | `us-central1` |
| Default table expiration | Never |

Or via CLI:
```
bq mk --location=us-central1 --dataset codelab-2-485215:ma_oracle
```

---

## Step 3 — Create the audit_log table

Run this DDL in **BigQuery Studio → SQL workspace**:

```sql
CREATE TABLE `codelab-2-485215.ma_oracle.audit_log` (
  query_id            STRING    NOT NULL,
  tenant_id           STRING    NOT NULL,
  user_id             STRING    NOT NULL,
  timestamp           TIMESTAMP NOT NULL,
  query               STRING    NOT NULL,
  route_taken         STRING,
  sources_retrieved   ARRAY<STRING>,
  retrieval_scores    ARRAY<FLOAT64>,
  graph_paths_traversed ARRAY<STRING>,
  generated_answer    STRING,
  confidence_score    FLOAT64,
  tokens_used         INT64,
  latency_ms          INT64,
  user_feedback       STRING,
  plan_type           STRING,
  cache_hit           BOOL,
  company             STRING,
  period              STRING
)
PARTITION BY DATE(timestamp)
CLUSTER BY tenant_id, route_taken
OPTIONS (
  description = 'M&A Oracle query audit log — one row per query',
  require_partition_filter = FALSE
);
```

> **Why partitioned by DATE(timestamp)?** Compliance queries are almost always
> time-bounded ("show me all queries from last month"). Partitioning cuts cost
> dramatically — BigQuery charges per bytes scanned.
>
> **Why clustered by tenant_id + route_taken?** The two most common filter
> patterns in multi-tenant audit reviews.

---

## Step 4 — Grant the application IAM access

The service account (or developer account for local dev) that runs the application
needs permission to write to BigQuery:

**GCP Console → IAM & Admin → IAM → Grant Access**

| Principal | Role |
|---|---|
| Developer Gmail / service account email | `BigQuery Data Editor` |
| Developer Gmail / service account email | `BigQuery Job User` |

Or via CLI (replace `PRINCIPAL` with the email):
```
gcloud projects add-iam-policy-binding codelab-2-485215 \
  --member="user:PRINCIPAL" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding codelab-2-485215 \
  --member="user:PRINCIPAL" \
  --role="roles/bigquery.jobUser"
```

---

## Step 5 — Verify table exists

```
bq show --schema codelab-2-485215:ma_oracle.audit_log
```

Or run a test query in BigQuery Studio:
```sql
SELECT COUNT(*) FROM `codelab-2-485215.ma_oracle.audit_log`;
-- Returns 0 rows initially — that's correct.
```

---

## Useful compliance queries

```sql
-- All queries by tenant in the last 7 days
SELECT query_id, user_id, query, route_taken, latency_ms, timestamp
FROM `codelab-2-485215.ma_oracle.audit_log`
WHERE tenant_id = 'tenant-abc'
  AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
ORDER BY timestamp DESC;

-- Route distribution
SELECT route_taken, COUNT(*) AS query_count, AVG(latency_ms) AS avg_latency_ms
FROM `codelab-2-485215.ma_oracle.audit_log`
GROUP BY route_taken
ORDER BY query_count DESC;

-- High-contradiction queries
SELECT query_id, query, company, period, generated_answer
FROM `codelab-2-485215.ma_oracle.audit_log`
WHERE route_taken = 'contradiction'
  AND confidence_score < 0.4
ORDER BY timestamp DESC;

-- Cache hit rate
SELECT
  COUNTIF(cache_hit) AS cache_hits,
  COUNT(*) AS total,
  ROUND(COUNTIF(cache_hit) / COUNT(*) * 100, 1) AS hit_rate_pct
FROM `codelab-2-485215.ma_oracle.audit_log`;
```
