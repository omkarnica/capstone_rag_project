"""
Build, summarize, embed, and upsert the RAPTOR tree for filing chunks.

Calling module:

    src.filings.pipeline.run_filings_pipeline()
      |
      v
    run_raptor_pipeline()

RAPTOR flow:

    chunks JSON
      |
      |-- load_chunks_from_json()
      |-- derive_source_doc_id()
      |-- build_raptor_tree()
      |     |-- embed_texts()
      |     |-- build_raptor_level()
      |     `-- summarize_cluster()
      |-- prepare_pinecone_vectors()
      |-- upsert_in_batches()
      `-- save_tree_map()

Runtime services:

    Pinecone provides embeddings and vector storage. Gemini provides summary
    generation. GaussianMixture performs clustering.
"""

import os
import json
import uuid
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from sklearn.mixture import GaussianMixture
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from google import genai

from .config_loader import load_config_yaml

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
_CONFIG = load_config_yaml(BASE_DIR / "config.yaml")


def _cfg_str(key: str, default: str) -> str:
    raw = str(_CONFIG.get(key, default)).strip()
    if "=" in raw:
        raw = raw.split("=", 1)[1].strip()
    return raw.strip('"').strip("'")


def _cfg_bool(key: str, default: bool = False) -> bool:
    raw = str(_CONFIG.get(key, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


GEMINI_PROJECT_ID_ENV = _cfg_str("GEMINI_PROJECT_ID_ENV", "GEMINI_PROJECT_ID")
GEMINI_LOCATION_ENV = _cfg_str("GEMINI_LOCATION_ENV", "GEMINI_LOCATION")
PINECONE_API_KEY_ENV = _cfg_str("PINECONE_API_KEY_ENV", "PINECONE_API_KEY")
PINECONE_INDEX_NAME = _cfg_str("PINECONE_INDEX_NAME", "ragcapstone")
PINECONE_NAMESPACE = _cfg_str("PINECONE_NAMESPACE", "apple")
RAPTOR_EMBED_MODEL = _cfg_str("RAPTOR_EMBED_MODEL", "all-MiniLM-L6-v2")
PINECONE_EMBED_MODEL = _cfg_str("PINECONE_EMBED_MODEL", "llama-text-embed-v2")
RAPTOR_SUMMARY_MODEL = _cfg_str("RAPTOR_SUMMARY_MODEL", "gemini-2.5-flash")
RAPTOR_SUMMARY_PROVIDER = _cfg_str("RAPTOR_SUMMARY_PROVIDER", "gemini").lower()
RAPTOR_GEMINI_MODE = _cfg_str("RAPTOR_GEMINI_MODE", "vertex").lower()
RAPTOR_EMBED_BATCH_SIZE = int(_cfg_str("RAPTOR_EMBED_BATCH_SIZE", "96"))
RAPTOR_GEMINI_COOLDOWN_SECONDS = float(_cfg_str("RAPTOR_GEMINI_COOLDOWN_SECONDS", "12"))
RAPTOR_SUMMARY_FALLBACK_MODELS = [
    m.strip()
    for m in _cfg_str(
        "RAPTOR_SUMMARY_FALLBACK_MODELS",
        "gemini-2.0-flash,gemini-1.5-flash,gemini-1.5-pro",
    ).split(",")
    if m.strip()
]

_PINECONE_CLIENT = None
_PINECONE_INDEX = None
_VERTEX_GENAI_CLIENT = None
_LAST_GEMINI_CALL_TS = 0.0
_GEMINI_MAX_RETRIES = 3
logger = logging.getLogger("raptor")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


def _get_vertex_genai_client() -> genai.Client:
    global _VERTEX_GENAI_CLIENT
    if _VERTEX_GENAI_CLIENT is not None:
        return _VERTEX_GENAI_CLIENT

    project_id = os.getenv(GEMINI_PROJECT_ID_ENV, "").strip()
    location = os.getenv(GEMINI_LOCATION_ENV, "").strip()
    if not project_id or not location:
        raise ValueError(
            f"{GEMINI_PROJECT_ID_ENV} or {GEMINI_LOCATION_ENV} is missing. "
            f"Set them in {BASE_DIR / '.env'} for Vertex Gemini calls."
        )

    _VERTEX_GENAI_CLIENT = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )
    return _VERTEX_GENAI_CLIENT


def _apply_gemini_cooldown() -> None:
    global _LAST_GEMINI_CALL_TS
    cooldown = max(0.0, RAPTOR_GEMINI_COOLDOWN_SECONDS)
    if cooldown <= 0:
        return
    elapsed = time.time() - _LAST_GEMINI_CALL_TS
    wait_seconds = cooldown - elapsed
    if wait_seconds > 0:
        logger.info("Gemini cooldown active: sleeping %.2f seconds", wait_seconds)
        time.sleep(wait_seconds)
    _LAST_GEMINI_CALL_TS = time.time()


def summarize_with_gemini_rest(prompt: str) -> str:
    model_candidates: List[str] = []
    for model_name in [RAPTOR_SUMMARY_MODEL, *RAPTOR_SUMMARY_FALLBACK_MODELS]:
        if model_name not in model_candidates:
            model_candidates.append(model_name)

    if RAPTOR_GEMINI_MODE != "vertex":
        raise ValueError(
            f"Unsupported RAPTOR_GEMINI_MODE='{RAPTOR_GEMINI_MODE}'. Use 'vertex'."
        )

    auth_modes = ["vertex"]
    last_error: Exception | None = None
    client = _get_vertex_genai_client()
    for auth_mode in auth_modes:
        for model_name in model_candidates:
            for attempt in range(_GEMINI_MAX_RETRIES + 1):
                _apply_gemini_cooldown()
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    summary_text = (getattr(response, "text", "") or "").strip()
                    if summary_text:
                        return summary_text
                    raise RuntimeError(f"Gemini returned empty text for model='{model_name}'")
                except Exception as exc:
                    last_error = exc
                    error_text = str(exc).lower()
                    is_last_attempt = attempt >= _GEMINI_MAX_RETRIES
                    is_not_found = "404" in error_text or "not found" in error_text
                    is_retryable = any(code in error_text for code in ["429", "500", "502", "503", "504"])

                    if is_not_found:
                        logger.warning(
                            "Gemini model not found auth_mode='%s' model='%s'. Trying next fallback. error=%s",
                            auth_mode,
                            model_name,
                            str(exc),
                        )
                        break

                    if is_retryable and not is_last_attempt:
                        sleep_seconds = min(2 ** attempt, 8)
                        logger.warning(
                            "Gemini transient SDK error auth_mode='%s' model='%s' attempt=%s. Retrying in %ss. error=%s",
                            auth_mode,
                            model_name,
                            attempt + 1,
                            sleep_seconds,
                            str(exc),
                        )
                        time.sleep(sleep_seconds)
                        continue

                    if not is_last_attempt:
                        logger.warning(
                            "Gemini SDK error auth_mode='%s' model='%s' attempt=%s. Retrying in 1s. error=%s",
                            auth_mode,
                            model_name,
                            attempt + 1,
                            str(exc),
                        )
                        time.sleep(1)
                        continue

                    logger.error(
                        "Gemini request failed auth_mode='%s' model='%s' attempts=%s error=%s",
                        auth_mode,
                        model_name,
                        attempt + 1,
                        str(exc),
                    )
                    break

    if last_error is not None:
        raise RuntimeError(
            f"Gemini failed for all auth modes {auth_modes} and models {model_candidates}. Last error: {last_error}"
        )
    raise RuntimeError(
        f"Gemini failed for all auth modes {auth_modes} and models {model_candidates}. "
        "Check model availability in region, key/token validity, and permissions."
    )


def get_pinecone_client():
    global _PINECONE_CLIENT
    if _PINECONE_CLIENT is not None:
        return _PINECONE_CLIENT
    pinecone_api_key = os.getenv(PINECONE_API_KEY_ENV, "").strip()
    if not pinecone_api_key:
        raise ValueError(
            f"{PINECONE_API_KEY_ENV} is missing. Set it in {BASE_DIR / '.env'}"
        )
    _PINECONE_CLIENT = Pinecone(api_key=pinecone_api_key)
    return _PINECONE_CLIENT


def get_pinecone_index():
    global _PINECONE_INDEX
    if _PINECONE_INDEX is not None:
        return _PINECONE_INDEX
    pc = get_pinecone_client()
    _PINECONE_INDEX = pc.Index(PINECONE_INDEX_NAME)
    return _PINECONE_INDEX


# =========================
# LLM SUMMARY
# =========================
def summarize_cluster(texts: List[str], max_chunks: int = 10) -> str:
    """
    Summarize a cluster of related chunks into one paragraph.
    """
    combined = "\n\n".join(texts[:max_chunks])

    system_prompt = (
        "You are summarizing a cluster of semantically related financial document passages. "
        "Write one coherent paragraph that preserves key facts, metrics, entities, and themes."
    )
    user_prompt = f"Summarize these related passages into one coherent paragraph:\n\n{combined}"

    if RAPTOR_SUMMARY_PROVIDER == "gemini":
        return summarize_with_gemini_rest(f"{system_prompt}\n\n{user_prompt}")

    raise ValueError(
        f"Unsupported RAPTOR_SUMMARY_PROVIDER='{RAPTOR_SUMMARY_PROVIDER}'. "
        "Use 'gemini'."
    )


# =========================
# EMBEDDING
# =========================
def embed_texts(embedder, texts: List[str]) -> np.ndarray:
    """
    Uses your local embedder, e.g. SentenceTransformer.
    Returns numpy array.
    """
    if not texts:
        return np.array([])

    # Primary path: Pinecone inference embeddings (e.g. llama-text-embed-v2).
    # This keeps RAPTOR vector dimension aligned with the Pinecone index.
    if embedder is None:
        pc = get_pinecone_client()
        batch_size = max(1, min(RAPTOR_EMBED_BATCH_SIZE, 96))
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch_inputs = texts[i:i + batch_size]
            response = pc.inference.embed(
                model=PINECONE_EMBED_MODEL,
                inputs=batch_inputs,
                parameters={"input_type": "passage", "truncate": "END"},
            )
            vectors.extend(item.values for item in response)
        return np.array(vectors, dtype=float)

    # Fallback path: local SentenceTransformer embedder.
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.array(embeddings)


# =========================
# RAPTOR NODE BUILDING
# =========================
def make_node_id(level: int, idx: int, prefix: str = "node") -> str:
    return f"{prefix}_L{level}_{idx}_{uuid.uuid4().hex[:8]}"


def _pick_cluster_metadata_value(cluster_members: List[Dict[str, Any]], key: str) -> Any:
    """
    Pick the first non-empty metadata value for a key from cluster members.
    Supports both leaf chunk_metadata and higher-level summary fields.
    """
    for member in cluster_members:
        chunk_md = member.get("chunk_metadata", {})
        if isinstance(chunk_md, dict):
            value = chunk_md.get(key)
            if value not in (None, ""):
                return value
        value = member.get(key)
        if value not in (None, ""):
            return value
    return None


def _to_int_or_none(value: Any) -> Union[int, None]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _derive_leaf_source_doc_id(chunk_metadata: Dict[str, Any], default_source_doc_id: str) -> str:
    """
    Prefer per-chunk source identity from metadata.
    Fallback order: source -> filename stem -> provided default_source_doc_id.
    """
    if isinstance(chunk_metadata, dict):
        source = str(chunk_metadata.get("source", "")).strip()
        if source:
            return source

        filename = str(chunk_metadata.get("filename", "")).strip()
        if filename:
            stem = Path(filename).stem.strip()
            if stem:
                return stem

    return default_source_doc_id


def build_raptor_level(
    current_nodes: List[Dict[str, Any]],
    embeddings: np.ndarray,
    level: int,
    n_components: int = 10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build one RAPTOR level using GMM clustering and summarization.

    current_nodes: list of nodes from previous level
    embeddings: embeddings of previous level texts
    level: new level number being created

    Returns:
        summary_nodes: new summary nodes for this level
        edges: mapping info between summary node and child node IDs
    """
    if len(current_nodes) < 2:
        return [], []

    # Avoid invalid number of clusters
    n_components = min(n_components, max(1, len(current_nodes) // 2))

    if n_components < 1:
        return [], []

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42
    )
    gmm.fit(embeddings)
    labels = gmm.predict(embeddings)

    summary_nodes = []
    edges = []

    summary_idx = 0
    for cluster_id in range(n_components):
        cluster_members = [
            current_nodes[i] for i, label in enumerate(labels) if label == cluster_id
        ]

        if not cluster_members:
            continue

        cluster_texts = [node["text"] for node in cluster_members]
        summary_text = summarize_cluster(cluster_texts)

        summary_node_id = make_node_id(level=level, idx=summary_idx, prefix="summary")
        child_ids = [node["id"] for node in cluster_members]
        summary_form_type = _pick_cluster_metadata_value(cluster_members, "form_type")
        summary_filename = _pick_cluster_metadata_value(cluster_members, "filename")
        summary_year = _pick_cluster_metadata_value(cluster_members, "year")

        summary_node = {
            "id": summary_node_id,
            "text": summary_text,
            "level": level,
            "node_type": "summary",
            "child_ids": child_ids,
            "parent_id": None,  # can be filled later if this node gets clustered upward
            "source_doc_id": cluster_members[0].get("source_doc_id"),
            "form_type": str(summary_form_type) if summary_form_type not in (None, "") else "",
            "filename": str(summary_filename) if summary_filename not in (None, "") else "",
            "year": _to_int_or_none(summary_year),
        }

        summary_nodes.append(summary_node)

        for child in cluster_members:
            child["parent_id"] = summary_node_id

        edges.append({
            "parent_id": summary_node_id,
            "child_ids": child_ids,
            "level": level,
        })

        summary_idx += 1

    logger.info(
        "Level %s completed: summary_nodes_created=%s, candidate_clusters=%s, input_nodes=%s",
        level,
        len(summary_nodes),
        n_components,
        len(current_nodes),
    )
    return summary_nodes, edges


# =========================
# FULL RAPTOR TREE
# =========================
def build_raptor_tree(
    leaf_chunks: List[Union[str, Dict[str, Any]]],
    embedder,
    source_doc_id: str,
    max_levels: int = 3,
    n_components: int = 10,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Builds RAPTOR tree from leaf chunks upward.

    Returns:
        all_nodes: leaves + summaries from all levels
        tree_map: structure for later retrieval/expansion
    """
    # Create leaf nodes from either:
    # - plain strings
    # - chunk objects from chunking_1.py: {"id","text","metadata"}
    leaf_nodes = []
    for i, chunk in enumerate(leaf_chunks):
        if isinstance(chunk, dict):
            chunk_text = str(chunk.get("text", "")).strip()
            chunk_id = str(chunk.get("id") or make_node_id(level=0, idx=i, prefix="leaf"))
            chunk_metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
        else:
            chunk_text = str(chunk).strip()
            chunk_id = make_node_id(level=0, idx=i, prefix="leaf")
            chunk_metadata = {}

        if not chunk_text:
            continue

        leaf_form_type = str(chunk_metadata.get("form_type", "")).strip() if isinstance(chunk_metadata, dict) else ""
        leaf_filename = str(chunk_metadata.get("filename", "")).strip() if isinstance(chunk_metadata, dict) else ""
        leaf_year = _to_int_or_none(chunk_metadata.get("year")) if isinstance(chunk_metadata, dict) else None

        leaf_nodes.append({
            "id": chunk_id,
            "text": chunk_text,
            "level": 0,
            "node_type": "leaf",
            "child_ids": [],
            "parent_id": None,
            "source_doc_id": _derive_leaf_source_doc_id(chunk_metadata, source_doc_id),
            "form_type": leaf_form_type,
            "filename": leaf_filename,
            "year": leaf_year,
            "chunk_metadata": chunk_metadata,
        })

    if not leaf_nodes:
        raise ValueError("No valid leaf chunks found. Ensure chunk text is non-empty.")

    all_nodes = list(leaf_nodes)
    tree_edges = []

    current_nodes = leaf_nodes
    current_embeddings = embed_texts(embedder, [n["text"] for n in current_nodes])
    total_summary_nodes_created = 0

    for level in range(1, max_levels + 1):
        if len(current_nodes) < 4:
            logger.info(
                "Stopping at level %s: current_nodes=%s (<4).",
                level,
                len(current_nodes),
            )
            break

        summary_nodes, edges = build_raptor_level(
            current_nodes=current_nodes,
            embeddings=current_embeddings,
            level=level,
            n_components=n_components,
        )

        if not summary_nodes:
            logger.info("Stopping at level %s: no summary nodes created.", level)
            break

        all_nodes.extend(summary_nodes)
        tree_edges.extend(edges)
        total_summary_nodes_created += len(summary_nodes)
        logger.info(
            "Level %s: added_summary_nodes=%s, running_total_summary_nodes=%s",
            level,
            len(summary_nodes),
            total_summary_nodes_created,
        )

        current_nodes = summary_nodes
        current_embeddings = embed_texts(embedder, [n["text"] for n in current_nodes])

    tree_map = {
        "source_doc_id": source_doc_id,
        "nodes": {
            node["id"]: {
                "text": node["text"],
                "level": node["level"],
                "node_type": node["node_type"],
                "child_ids": node["child_ids"],
                "parent_id": node["parent_id"],
                "form_type": node.get("form_type", ""),
                "filename": node.get("filename", ""),
                "year": node.get("year"),
            }
            for node in all_nodes
        },
        "edges": tree_edges,
    }
    logger.info(
        "RAPTOR tree build complete: leaf_nodes=%s, total_summary_nodes=%s, total_nodes=%s, levels_attempted=%s",
        len(leaf_nodes),
        total_summary_nodes_created,
        len(all_nodes),
        max_levels,
    )

    return all_nodes, tree_map


# =========================
# PINECONE UPSERT
# =========================
def prepare_pinecone_vectors(nodes: List[Dict[str, Any]], embedder) -> List[Dict[str, Any]]:
    """
    Convert RAPTOR nodes into Pinecone upsert format.
    """
    texts = [node["text"] for node in nodes]
    embeddings = embed_texts(embedder, texts)

    vectors = []
    for node, emb in zip(nodes, embeddings):
        metadata = {
            "text": node["text"][:35000],  # keep metadata safe
            "node_type": node["node_type"],
            "level": int(node["level"]),
            "source_doc_id": node.get("source_doc_id", ""),
            "parent_id": node.get("parent_id", "") or "",
        }
        if node.get("node_type") == "summary":
            metadata["child_ids"] = [str(child_id) for child_id in node.get("child_ids", [])]
            metadata["child_count"] = len(metadata["child_ids"])
            metadata["form_type"] = str(node.get("form_type", ""))
            metadata["filename"] = str(node.get("filename", ""))
            metadata["year"] = _to_int_or_none(node.get("year"))
            metadata = {k: v for k, v in metadata.items() if v is not None}

        if node.get("node_type") == "leaf":
            # Preserve useful chunk-level metadata from chunking_1.py.
            chunk_md = node.get("chunk_metadata", {})
            if isinstance(chunk_md, dict):
                metadata.update({
                    "chunk_id": str(chunk_md.get("chunk_id", node.get("id", ""))),
                    "ticker": str(chunk_md.get("ticker", "")),
                    "year": int(chunk_md["year"]) if chunk_md.get("year") not in (None, "") else None,
                    "form_type": str(chunk_md.get("form_type", "")),
                    "company_title": str(chunk_md.get("company_title", ""))[:500],
                    "filename": str(chunk_md.get("filename", "")),
                    "part_title": str(chunk_md.get("part_title", ""))[:500],
                    "item_title": str(chunk_md.get("item_title", ""))[:500],
                    "note_title": str(chunk_md.get("note_title", ""))[:500],
                    "subsection_title": str(chunk_md.get("subsection_title", ""))[:500],
                    "block_index": int(chunk_md["block_index"]) if chunk_md.get("block_index") not in (None, "") else None,
                    "chunk_index_within_block": int(chunk_md["chunk_index_within_block"]) if chunk_md.get("chunk_index_within_block") not in (None, "") else None,
                    "content_type": str(chunk_md.get("content_type", "")),
                    "is_front_matter": bool(chunk_md.get("is_front_matter", False)),
                })
                metadata = {k: v for k, v in metadata.items() if v is not None}

        vectors.append({
            "id": node["id"],
            "values": emb.tolist(),
            "metadata": metadata,
        })

    return vectors


def upsert_in_batches(index, vectors: List[Dict[str, Any]], namespace: str, batch_size: int = 100):
    """
    Upsert vectors to Pinecone in batches.
    """
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)


# =========================
# SAVE TREE MAP
# =========================
def save_tree_map(tree_map: Dict[str, Any], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree_map, f, indent=2, ensure_ascii=False)


# =========================
# MAIN PIPELINE
# =========================
def process_and_upsert_raptor(
    leaf_chunks: List[Union[str, Dict[str, Any]]],
    embedder,
    source_doc_id: str,
    namespace: str = "default",
    max_levels: int = 3,
    n_components: int = 10,
    tree_output_path: str = "raptor_tree.json",
):
    """
    End-to-end:
    1. Build RAPTOR tree
    2. Embed all nodes
    3. Upsert to Pinecone
    4. Save tree map locally
    """
    all_nodes, tree_map = build_raptor_tree(
        leaf_chunks=leaf_chunks,
        embedder=embedder,
        source_doc_id=source_doc_id,
        max_levels=max_levels,
        n_components=n_components,
    )

    vectors = prepare_pinecone_vectors(all_nodes, embedder)
    upsert_in_batches(index=get_pinecone_index(), vectors=vectors, namespace=namespace, batch_size=100)
    save_tree_map(tree_map, tree_output_path)

    logger.info("Upserted %s RAPTOR nodes to Pinecone namespace='%s'", len(vectors), namespace)
    logger.info("Saved RAPTOR tree map to: %s", tree_output_path)

    return all_nodes, tree_map


def load_chunks_from_json(chunks_json_path: str) -> List[Dict[str, Any]]:
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Chunks JSON must be a list of chunk objects.")

    normalized = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        normalized.append({
            "id": str(item.get("id") or metadata.get("chunk_id") or make_node_id(level=0, idx=i, prefix="leaf")),
            "text": text,
            "metadata": metadata,
        })

    return normalized


def derive_source_doc_id(chunks: List[Dict[str, Any]], default_value: str = "finance_doc") -> str:
    """
    Prefer chunk metadata fields generated by chunking_1.py:
    source -> filename -> ticker/year/form_type composite.
    """
    if not chunks:
        return default_value

    first_md = chunks[0].get("metadata", {}) if isinstance(chunks[0], dict) else {}
    if not isinstance(first_md, dict):
        return default_value

    source = str(first_md.get("source", "")).strip()
    if source:
        return source

    filename = str(first_md.get("filename", "")).strip()
    if filename:
        stem = Path(filename).stem.strip()
        if stem:
            return stem

    ticker = str(first_md.get("ticker", "")).strip().upper()
    year = str(first_md.get("year", "")).strip()
    form_type = str(first_md.get("form_type", "")).strip().upper()

    parts = [p for p in [ticker, year, form_type] if p]
    if parts:
        return "_".join(parts)

    return default_value


def derive_tree_output_filename(
    chunks: List[Dict[str, Any]],
    default_stem: str = "finance_doc",
) -> str:
    """
    Save RAPTOR tree as ticker_form_type_raptor_tree.json when possible.
    """
    ticker = ""
    form_type = ""
    if chunks and isinstance(chunks[0], dict):
        first_md = chunks[0].get("metadata", {})
        if isinstance(first_md, dict):
            ticker = str(first_md.get("ticker", "")).strip().lower()
            form_type = str(first_md.get("form_type", "")).strip().lower()

    if ticker and form_type:
        return f"{ticker}_{form_type}_raptor_tree.json"

    return f"{default_stem}_raptor_tree.json"


def run_main_self_test(chunks: List[Dict[str, Any]], embedder, source_doc_id: str) -> None:
    """
    Quick local sanity checks before full upsert.
    Uses max_levels=0 to avoid LLM summarization calls in test mode.
    """
    if not chunks:
        raise ValueError("Self-test failed: no chunks loaded.")
    if not source_doc_id:
        raise ValueError("Self-test failed: source_doc_id is empty.")

    sample_size = min(8, len(chunks))
    sample_chunks = chunks[:sample_size]

    test_nodes, test_tree = build_raptor_tree(
        leaf_chunks=sample_chunks,
        embedder=embedder,
        source_doc_id=source_doc_id,
        max_levels=0,
        n_components=2,
    )
    if len(test_nodes) != sample_size:
        raise AssertionError(
            f"Self-test failed: expected {sample_size} leaf nodes, got {len(test_nodes)}."
        )

    test_vectors = prepare_pinecone_vectors(test_nodes, embedder)
    if len(test_vectors) != len(test_nodes):
        raise AssertionError("Self-test failed: vector count does not match node count.")

    logger.info("[SELF-TEST] PASS")
    logger.info("[SELF-TEST] Sample chunks: %s", sample_size)
    logger.info("[SELF-TEST] Nodes built: %s", len(test_nodes))
    logger.info("[SELF-TEST] Tree edges: %s", len(test_tree.get("edges", [])))
    logger.info("[SELF-TEST] Vectors prepared: %s", len(test_vectors))


def run_raptor_pipeline(
    chunks_json_path: str | None = None,
    namespace: str | None = None,
    embedder=None,
    max_levels: int = 3,
    n_components: int = 10,
    run_self_test: bool = True,
    self_test_only: bool | None = None,
):
    """
    Final end-to-end RAPTOR pipeline wrapper.

    Flow:
    1) Load chunks
    2) Derive source_doc_id + output filename
    3) Optional self-test
    4) Build RAPTOR tree + upsert + save tree map
    """
    chunks_path = Path(chunks_json_path) if chunks_json_path else (BASE_DIR / "aapl_8-k_pinecone_ready_chunks.json")
    namespace_value = namespace or PINECONE_NAMESPACE
    self_test_only_value = (
        os.getenv("RAPTOR_SELF_TEST_ONLY", "false").strip().lower() == "true"
        if self_test_only is None
        else bool(self_test_only)
    )

    chunks = load_chunks_from_json(str(chunks_path))
    source_doc_id = derive_source_doc_id(chunks, default_value=chunks_path.stem)
    tree_output_filename = derive_tree_output_filename(chunks, default_stem=source_doc_id)

    if run_self_test:
        run_main_self_test(chunks=chunks, embedder=embedder, source_doc_id=source_doc_id)

    if self_test_only_value:
        logger.info("[SELF-TEST] RAPTOR_SELF_TEST_ONLY=true, skipping Pinecone upsert.")
        return None

    return process_and_upsert_raptor(
        leaf_chunks=chunks,
        embedder=embedder,
        source_doc_id=source_doc_id,
        namespace=namespace_value,
        max_levels=max_levels,
        n_components=n_components,
        tree_output_path=tree_output_filename,
    )
