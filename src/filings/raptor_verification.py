"""
Validate a saved RAPTOR tree map after indexing.

Calling module:

    src.filings.pipeline.run_filings_pipeline()
      |
      v
    verify_raptor_tree()

Verification flow:

    RAPTOR tree JSON
      |
      |-- load_tree()
      |-- check_structure()
      |-- detect_cycles()
      |-- inspect_sample_summaries()
      `-- get_pinecone_namespace_count() when an index is supplied

Primary output:

    Dict with node counts, structural issues, cycle checks, Pinecone count,
    and ready_for_retrieval status.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

from dotenv import load_dotenv

from .config_loader import load_config_yaml

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
_CONFIG = load_config_yaml(BASE_DIR / "config.yaml")
logger = logging.getLogger("raptor_verification")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _clean_config_str(value: Any, default: str) -> str:
    raw = str(value if value is not None else default).strip()
    if "=" in raw:
        raw = raw.split("=", 1)[1].strip()
    return raw.strip('"').strip("'")


PINECONE_API_KEY_ENV = _clean_config_str(_CONFIG.get("PINECONE_API_KEY_ENV"), "PINECONE_API_KEY")
PINECONE_INDEX_NAME = _clean_config_str(_CONFIG.get("PINECONE_INDEX_NAME"), "ragcapstone")
PINECONE_NAMESPACE = _clean_config_str(_CONFIG.get("PINECONE_NAMESPACE"), "user1")


def load_tree(tree_path: str) -> Dict[str, Any]:
    with open(tree_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_structure(nodes: Dict[str, Any]) -> Dict[str, Any]:
    level_counts = defaultdict(int)
    type_counts = defaultdict(int)

    missing_children = []
    bad_parent_links = []
    empty_summary_nodes = []
    leaf_with_children = []
    roots = []
    leaves = []
    summaries = []

    for node_id, node in nodes.items():
        node_type = node.get("node_type")
        level = node.get("level")
        child_ids = node.get("child_ids", [])
        parent_id = node.get("parent_id")

        level_counts[level] += 1
        type_counts[node_type] += 1

        if parent_id is None:
            roots.append(node_id)

        if node_type == "leaf":
            leaves.append(node_id)
            if child_ids:
                leaf_with_children.append(node_id)

        elif node_type == "summary":
            summaries.append(node_id)
            if not child_ids:
                empty_summary_nodes.append(node_id)

        for child_id in child_ids:
            if child_id not in nodes:
                missing_children.append((node_id, child_id))
            else:
                actual_parent = nodes[child_id].get("parent_id")
                if actual_parent != node_id:
                    bad_parent_links.append((node_id, child_id, actual_parent))

    return {
        "level_counts": dict(sorted(level_counts.items())),
        "type_counts": dict(type_counts),
        "roots": roots,
        "leaves": leaves,
        "summaries": summaries,
        "missing_children": missing_children,
        "bad_parent_links": bad_parent_links,
        "empty_summary_nodes": empty_summary_nodes,
        "leaf_with_children": leaf_with_children,
    }


def detect_cycles(nodes: Dict[str, Any]) -> List[List[str]]:
    visited: Set[str] = set()
    stack: Set[str] = set()
    path: List[str] = []
    cycles: List[List[str]] = []

    def dfs(node_id: str):
        if node_id in stack:
            if node_id in path:
                cycle_start = path.index(node_id)
                cycles.append(path[cycle_start:] + [node_id])
            return

        if node_id in visited:
            return

        visited.add(node_id)
        stack.add(node_id)
        path.append(node_id)

        for child_id in nodes[node_id].get("child_ids", []):
            if child_id in nodes:
                dfs(child_id)

        path.pop()
        stack.remove(node_id)

    for node_id in nodes:
        if node_id not in visited:
            dfs(node_id)

    return cycles


def inspect_sample_summaries(nodes: Dict[str, Any], max_samples: int = 3) -> List[Dict[str, Any]]:
    samples = []
    summary_nodes = [(nid, n) for nid, n in nodes.items() if n.get("node_type") == "summary"]

    for node_id, node in summary_nodes[:max_samples]:
        samples.append({
            "id": node_id,
            "level": node.get("level"),
            "num_children": len(node.get("child_ids", [])),
            "text_preview": node.get("text", "")[:300]
        })

    return samples


def get_pinecone_namespace_count(index, namespace: str):
    """
    Tries to read Pinecone stats for the given namespace.
    Returns count if available, else None.
    """
    if index is None:
        return None

    try:
        stats = index.describe_index_stats()
        # depending on SDK version, stats may be object-like or dict-like
        if hasattr(stats, "namespaces"):
            namespaces = stats.namespaces
            ns_info = namespaces.get(namespace)
            if ns_info is None:
                return 0
            if hasattr(ns_info, "vector_count"):
                return ns_info.vector_count
            return ns_info.get("vector_count", None)

        # dict-style fallback
        namespaces = stats.get("namespaces", {})
        ns_info = namespaces.get(namespace)
        if ns_info is None:
            return 0
        return ns_info.get("vector_count", None)

    except Exception as e:
        logger.info("Could not fetch Pinecone stats: %s", e)
        return None


def verify_raptor_tree(tree_path: str, index=None, namespace: str = "user1") -> Dict[str, Any]:
    tree = load_tree(tree_path)

    nodes = tree.get("nodes", {})
    edges = tree.get("edges", [])
    logger.info("Starting RAPTOR tree verification: %s", tree_path)

    structure = check_structure(nodes)
    cycles = detect_cycles(nodes)
    pinecone_count = get_pinecone_namespace_count(index, namespace)
    samples = inspect_sample_summaries(nodes, max_samples=3)

    # readiness rules
    ready = (
        len(nodes) > 0
        and len(structure["summaries"]) > 0
        and len(structure["leaves"]) > 0
        and len(structure["missing_children"]) == 0
        and len(structure["bad_parent_links"]) == 0
        and len(structure["empty_summary_nodes"]) == 0
        and len(structure["leaf_with_children"]) == 0
        and len(cycles) == 0
    )

    logger.info(
        "Verification summary: nodes=%s edges=%s leaves=%s summaries=%s cycles=%s missing_children=%s",
        len(nodes),
        len(edges),
        len(structure["leaves"]),
        len(structure["summaries"]),
        len(cycles),
        len(structure["missing_children"]),
    )
    if pinecone_count is not None:
        logger.info(
            "Pinecone namespace '%s' count=%s delta_vs_json=%s",
            namespace,
            pinecone_count,
            pinecone_count - len(nodes),
        )
    if ready:
        logger.info("STATUS: TREE LOOKS READY FOR RETRIEVAL")
    else:
        logger.info("STATUS: TREE NEEDS FIXES BEFORE RETRIEVAL")

    return {
        "tree_path": tree_path,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "level_counts": structure["level_counts"],
        "type_counts": structure["type_counts"],
        "roots": structure["roots"],
        "leaves": structure["leaves"],
        "summaries": structure["summaries"],
        "missing_children": structure["missing_children"],
        "bad_parent_links": structure["bad_parent_links"],
        "empty_summary_nodes": structure["empty_summary_nodes"],
        "leaf_with_children": structure["leaf_with_children"],
        "cycles": cycles,
        "pinecone_count": pinecone_count,
        "ready_for_retrieval": ready,
    }





