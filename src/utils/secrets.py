def get_secret(secret_id: str, project_id: str = "codelab-2-485215") -> str:
    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8").strip()
    except Exception:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        value = os.getenv(secret_id)
        if not value:
            raise ValueError(f"Secret {secret_id} not found in GCP or .env")
        return value
