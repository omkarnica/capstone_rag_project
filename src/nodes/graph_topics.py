from __future__ import annotations


GRAPH_MARKERS = {
    "board member",
    "board members",
    "board of directors",
    "board composition",
    "director",
    "directors",
    "subsidiary",
    "subsidiaries",
    "corporate hierarchy",
    "ownership structure",
    "ownership hierarchy",
}


def is_graph_topic(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in GRAPH_MARKERS)
