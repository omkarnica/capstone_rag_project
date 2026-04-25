"""
Embedding utilities for the cache system.
"""

import hashlib
import re

import numpy as np
from pinecone import Pinecone
from src.utils.secrets import get_secret

PINECONE_EMBED_MODEL = "llama-text-embed-v2"

_embeddings_client = None


def _get_embeddings_client():
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = Pinecone(api_key=get_secret("PINECONE_API_KEY"))
    return _embeddings_client


def embed_query(text: str) -> list[float]:
    pc = _get_embeddings_client()
    response = pc.inference.embed(
        model=PINECONE_EMBED_MODEL,
        inputs=[text],
        parameters={"input_type": "query"},
    )
    return response.data[0].values


def normalize_query(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[?.!]+$", "", text).strip()
    return text


def hash_query(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def embedding_to_bytes(embedding: list[float]) -> bytes:
    return np.array(embedding, dtype=np.float32).tobytes()


def bytes_to_embedding(data: bytes) -> list[float]:
    return np.frombuffer(data, dtype=np.float32).tolist()
