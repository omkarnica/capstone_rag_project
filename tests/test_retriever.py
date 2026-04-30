from __future__ import annotations

from unittest.mock import patch


def test_grade_documents_keeps_all_graph_board_member_rows_for_list_query():
    from src.nodes.grader import grade_documents

    docs = [
        {"content": "Apple board member in 2024: Alex Gorsky | Title: Former Chair and CEO, Johnson & Johnson", "metadata": {"source": "Knowledge Graph"}},
        {"content": "Apple board member in 2024: Andrea Jung | Title: President and CEO, Grameen America", "metadata": {"source": "Knowledge Graph"}},
        {"content": "Apple board member in 2024: Tim Cook | Title: CEO, Apple", "metadata": {"source": "Knowledge Graph"}},
    ]

    result = grade_documents(
        {
            "route": "graph",
            "question": "Who are the board members of Apple in 2024",
            "rewritten_question": "Apple Inc. board of directors 2024",
            "retrieved_docs": docs,
        }
    )

    assert result["doc_relevance"] == ["yes", "yes", "yes"]
    assert result["relevant_doc_count"] == 3
    assert result["filtered_docs"] == docs


def test_grade_documents_keeps_all_graph_subsidiary_rows_for_list_query():
    from src.nodes.grader import grade_documents

    docs = [
        {"content": "Apple subsidiary: Apple Operations International Limited", "metadata": {"source": "Knowledge Graph"}},
        {"content": "Apple subsidiary: Apple Sales International", "metadata": {"source": "Knowledge Graph"}},
    ]

    result = grade_documents(
        {
            "route": "graph",
            "question": "List Apple's subsidiaries",
            "rewritten_question": "Apple subsidiaries",
            "retrieved_docs": docs,
        }
    )

    assert result["doc_relevance"] == ["yes", "yes"]
    assert result["relevant_doc_count"] == 2


def test_non_graph_routes_still_use_llm_grading():
    from src.nodes.grader import BinaryGrade, grade_documents

    class StubGrader:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, _prompt):
            return BinaryGrade(score="no")

    docs = [
        {"content": "Some filing chunk", "metadata": {"source": "SEC Filing"}},
    ]

    with patch("src.nodes.grader.get_grader_llm", return_value=StubGrader()):
        result = grade_documents(
            {
                "route": "filings",
                "question": "What are Apple's risk factors?",
                "retrieved_docs": docs,
            }
        )

    assert result["doc_relevance"] == ["no"]
    assert result["relevant_doc_count"] == 0


def test_graph_routes_do_not_call_generic_llm_grader():
    from src.nodes.grader import grade_documents

    docs = [
        {"content": "Apple board member in 2024: Tim Cook | Title: CEO, Apple", "metadata": {"source": "Knowledge Graph"}},
    ]

    with patch("src.nodes.grader.get_grader_llm", side_effect=AssertionError("generic grader should not run for graph")):
        result = grade_documents(
            {
                "route": "graph",
                "question": "Who are the board members of Apple in 2024",
                "retrieved_docs": docs,
            }
        )

    assert result["doc_relevance"] == ["yes"]
    assert result["relevant_doc_count"] == 1


def test_retrieve_docs_marks_filings_failure():
    from src.nodes.retriever import retrieve_docs

    with patch("src.filings.raptor_retrieval.raptor_retrieve", side_effect=RuntimeError("boom")):
        result = retrieve_docs({"route": "filings", "question": "Summarize Apple's risk factors"})

    assert result["filings_error"] is True
    assert result["filings_empty"] is True
    assert result["retrieved_docs"] == []


def test_retrieve_docs_marks_filings_empty_result():
    from src.nodes.retriever import retrieve_docs

    with patch("src.filings.raptor_retrieval.raptor_retrieve", return_value={"contexts": []}):
        result = retrieve_docs({"route": "filings", "question": "Summarize Apple's risk factors"})

    assert result["filings_error"] is False
    assert result["filings_empty"] is True


def test_route_after_doc_grading_falls_back_to_graph_for_empty_filings():
    from src.graph import route_after_doc_grading

    state = {
        "route": "filings",
        "relevant_doc_count": 0,
        "retrieval_attempt": 0,
        "max_retrieval_attempts": 3,
        "filings_empty": True,
        "graph_fallback_attempted": False,
    }

    assert route_after_doc_grading(state) == "graph_fallback"


def test_route_after_doc_grading_falls_back_to_graph_for_filings_error():
    from src.graph import route_after_doc_grading

    state = {
        "route": "filings",
        "relevant_doc_count": 0,
        "retrieval_attempt": 0,
        "max_retrieval_attempts": 3,
        "filings_error": True,
        "graph_fallback_attempted": False,
    }

    assert route_after_doc_grading(state) == "graph_fallback"


def test_route_after_doc_grading_falls_back_to_graph_after_irrelevant_filings_docs():
    from src.graph import route_after_doc_grading

    state = {
        "route": "filings",
        "relevant_doc_count": 0,
        "retrieval_attempt": 0,
        "max_retrieval_attempts": 3,
        "filings_error": False,
        "filings_empty": False,
        "graph_fallback_attempted": False,
    }

    assert route_after_doc_grading(state) == "graph_fallback"


def test_route_after_doc_grading_only_falls_back_once():
    from src.graph import route_after_doc_grading

    state = {
        "route": "filings",
        "relevant_doc_count": 0,
        "retrieval_attempt": 0,
        "max_retrieval_attempts": 3,
        "filings_empty": True,
        "graph_fallback_attempted": True,
    }

    assert route_after_doc_grading(state) == "rewrite"


def test_retry_route_ends_for_graph_route():
    from src.graph import retry_route

    state = {
        "route": "graph",
    }

    assert retry_route(state) == "end"


def test_grade_documents_keeps_all_graph_board_member_rows_for_list_query():
    from src.nodes.grader import grade_documents

    docs = [
        {
            "content": "Apple board member in 2024: Alex Gorsky | Title: Former Chair and CEO, Johnson & Johnson",
            "metadata": {"source": "Knowledge Graph"},
        },
        {
            "content": "Apple board member in 2024: Andrea Jung | Title: President and CEO, Grameen America",
            "metadata": {"source": "Knowledge Graph"},
        },
        {
            "content": "Apple board member in 2024: Tim Cook | Title: CEO, Apple",
            "metadata": {"source": "Knowledge Graph"},
        },
    ]

    result = grade_documents(
        {
            "route": "graph",
            "question": "Who are the board members of Apple in 2024",
            "rewritten_question": "Apple Inc. board of directors 2024",
            "retrieved_docs": docs,
        }
    )

    assert result["doc_relevance"] == ["yes", "yes", "yes"]
    assert result["relevant_doc_count"] == 3
    assert result["filtered_docs"] == docs


def test_grade_documents_keeps_all_graph_subsidiary_rows_for_list_query():
    from src.nodes.grader import grade_documents

    docs = [
        {"content": "Apple subsidiary: Apple Operations International Limited", "metadata": {"source": "Knowledge Graph"}},
        {"content": "Apple subsidiary: Apple Sales International", "metadata": {"source": "Knowledge Graph"}},
    ]

    result = grade_documents(
        {
            "route": "graph",
            "question": "List Apple's subsidiaries",
            "rewritten_question": "Apple subsidiaries",
            "retrieved_docs": docs,
        }
    )

    assert result["doc_relevance"] == ["yes", "yes"]
    assert result["relevant_doc_count"] == 2
