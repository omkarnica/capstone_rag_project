from __future__ import annotations


def test_planner_hints_graph_for_board_member_query():
    from src.nodes.planner import _guess_route_hint

    assert _guess_route_hint("Who are Apple's board members?") == "graph"


def test_planner_hints_graph_for_subsidiary_query():
    from src.nodes.planner import _guess_route_hint

    assert _guess_route_hint("List Apple's subsidiaries") == "graph"
