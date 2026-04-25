# GCP Local Development Setup

How to authenticate your local machine against the `codelab-2-485215` GCP project
so that Secret Manager, Vertex AI (Gemini), and other services work without any
API keys or JSON credential files in the codebase.

---

## Prerequisites

- A Google account that has been granted IAM access to `codelab-2-485215`
  (see [Required IAM Roles](#required-iam-roles) — ask the project admin to add you)
- Google Cloud SDK installed on your machine

---

## Step 1 — Install Google Cloud SDK (Windows)

Download and run the installer from:
https://cloud.google.com/sdk/docs/install-sdk#windows

The installer adds `gcloud` to your PATH automatically.

Verify in a **new** CMD window:

```
gcloud version
```

Expected output (versions may differ):
```
Google Cloud SDK 565.0.0
bq 2.1.31
core 2026.04.10
```

> **Note:** If you use Git Bash or WSL, `gcloud` may not be in that shell's PATH even
> after installation. Run authentication steps in **CMD** or **PowerShell** instead.

---

## Step 2 — Authenticate with your Google account

```
gcloud auth login
```

This opens a browser. Sign in with the Google account that has been granted access
to the `codelab-2-485215` project (not a personal Gmail unless the admin has
explicitly added it).

Set the active project:

```
gcloud config set project codelab-2-485215
```

---

## Step 3 — Set up Application Default Credentials (ADC)

ADC is what Python libraries (`google-cloud-*`, `google-genai`, etc.) use to
authenticate automatically — no code changes needed.

```
gcloud auth application-default login
```

This opens a second browser flow. Sign in with the same account as Step 2.
On success you see: **"You are now authenticated with the gcloud CLI!"**

---

## Step 4 — Set the quota project

This tells GCP which project to bill API usage against:

```
gcloud auth application-default set-quota-project codelab-2-485215
```

> **If you see a PERMISSION_DENIED error here**, your account has not been granted
> the `serviceusage.services.use` permission yet. Contact the project admin —
> see [Required IAM Roles](#required-iam-roles) below.

---

## Step 5 — Verify everything works

Run from the project root:

```
uv run python -c "
import google.auth
creds, project = google.auth.default()
print('ADC OK — project:', project)

from src.utils.secrets import get_secret
key = get_secret('PINECONE_API_KEY')
print('Secret Manager OK — PINECONE_API_KEY:', key[:4] + '...' + key[-4:])

from src.model_config import get_genai_client, get_model_name
client = get_genai_client()
response = client.models.generate_content(
    model=get_model_name(),
    contents='In one sentence, what is M&A due diligence?'
)
print('Vertex AI OK —', response.text.strip())
"
```

All three lines should print without errors.

---

## Required IAM Roles

The project admin must grant these roles to each developer's Google account
in **GCP Console → IAM & Admin → IAM → Grant Access**:

| Role | Why needed |
|---|---|
| `Secret Manager Secret Accessor` | Read secrets from Secret Manager at runtime |
| `Vertex AI User` | Call Gemini models via Vertex AI |
| `Service Usage Consumer` | Set quota project for ADC (Step 4) |

For dev convenience the admin may grant `Editor` instead, but least-privilege
is the enterprise standard.

---

## Enterprise context

This setup mirrors how production access works at companies:

| Environment | Auth mechanism |
|---|---|
| Local dev | `gcloud auth application-default login` with a corporate Google account + IAM grant |
| Cloud Run (deployed) | Service account attached at deploy time — metadata server provides credentials automatically, no JSON files or env vars needed |
| Secrets | GCP Secret Manager fetched at startup via `src/utils/secrets.py:preload_secrets()` — never stored in `.env` or committed to git |

No API keys, no JSON credential files, no secrets in code — ever.

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `DefaultCredentialsError: credentials were not found` | ADC not set up | Run Step 3 |
| `PERMISSION_DENIED: serviceusage.services.use` | Account not in project IAM | Ask admin to grant roles (see above) |
| `403 PERMISSION_DENIED on aiplatform.endpoints.predict` | Quota project not set | Run Step 4 after admin grants access |
| `gcloud: command not found` (bash/WSL) | gcloud not in bash PATH | Run commands in CMD/PowerShell instead |
| `Secret X not found in GCP or .env` | Secret doesn't exist in Secret Manager | Ask admin to create it, or add to local `.env` for dev |
