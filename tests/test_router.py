from __future__ import annotations

from unittest.mock import patch


def test_router_routes_board_member_query_to_graph():
    from src.nodes.router import route_question

    with patch("src.nodes.router.get_router_llm", side_effect=AssertionError("LLM should not be used")):
        result = route_question({"question": "Who are Apple's board members?"})

    assert result["route"] == "graph"
    assert result["initial_route"] == "graph"


def test_router_routes_subsidiary_query_to_graph():
    from src.nodes.router import route_question

    with patch("src.nodes.router.get_router_llm", side_effect=AssertionError("LLM should not be used")):
        result = route_question({"question": "List Apple's subsidiaries"})

    assert result["route"] == "graph"
    assert result["initial_route"] == "graph"
