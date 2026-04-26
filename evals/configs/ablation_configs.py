from __future__ import annotations

ABLATION_CONFIGS: dict[str, dict] = {
    "naive_rag": {
        "router": False,
        "reranker": False,
        "corrective": False,
        "self_rag": False,
    },
    "plus_router": {
        "router": True,
        "reranker": False,
        "corrective": False,
        "self_rag": False,
    },
    "plus_reranker": {
        "router": True,
        "reranker": True,
        "corrective": False,
        "self_rag": False,
    },
    "full_system": {
        "router": True,
        "reranker": True,
        "corrective": True,
        "self_rag": True,
    },
    # Plug in when teammate's KG branch merges:
    # "plus_kg": {
    #     "router": True, "reranker": True, "corrective": True,
    #     "self_rag": True, "kg": True,
    # },
}
