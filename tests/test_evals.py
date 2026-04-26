from unittest.mock import MagicMock, patch


def test_gemini_judge_get_model_name():
    with patch("evals.metrics.gemini_judge.get_genai_client", return_value=MagicMock()):
        from evals.metrics.gemini_judge import GeminiJudge
        judge = GeminiJudge()
        assert judge.get_model_name() == "gemini-2.5-flash"


def test_gemini_judge_generate_calls_genai():
    mock_response = MagicMock()
    mock_response.text = "yes"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("evals.metrics.gemini_judge.get_genai_client", return_value=mock_client):
        from evals.metrics.gemini_judge import GeminiJudge
        judge = GeminiJudge()
        result = judge.generate("Is this faithful?")

    assert result == "yes"
    mock_client.models.generate_content.assert_called_once()


def test_mrr_perfect_match():
    from evals.metrics.retrieval import MRRMetric
    metric = MRRMetric()
    score = metric.compute(
        retrieved_sources=["AAPL 10-K 2024", "MSFT 10-K 2024"],
        expected_sources=["AAPL 10-K 2024"],
    )
    assert score == 1.0


def test_mrr_second_position():
    from evals.metrics.retrieval import MRRMetric
    metric = MRRMetric()
    score = metric.compute(
        retrieved_sources=["MSFT 10-K 2024", "AAPL 10-K 2024"],
        expected_sources=["AAPL 10-K 2024"],
    )
    assert score == 0.5


def test_mrr_no_match():
    from evals.metrics.retrieval import MRRMetric
    metric = MRRMetric()
    score = metric.compute(
        retrieved_sources=["MSFT 10-K 2024"],
        expected_sources=["AAPL 10-K 2024"],
    )
    assert score == 0.0


def test_ndcg_perfect():
    from evals.metrics.retrieval import NDCGMetric
    metric = NDCGMetric(k=3)
    score = metric.compute(
        retrieved_sources=["AAPL 10-K 2024", "AAPL XBRL FY2024", "MSFT 10-K 2024"],
        expected_sources=["AAPL 10-K 2024", "AAPL XBRL FY2024"],
    )
    assert score == 1.0


def test_ndcg_none_relevant():
    from evals.metrics.retrieval import NDCGMetric
    metric = NDCGMetric(k=3)
    score = metric.compute(
        retrieved_sources=["NVDA 10-K 2024"],
        expected_sources=["AAPL 10-K 2024"],
    )
    assert score == 0.0


def test_ndcg_relevant_at_rank_two():
    from evals.metrics.retrieval import NDCGMetric
    import math
    metric = NDCGMetric(k=3)
    score = metric.compute(
        retrieved_sources=["MSFT 10-K 2024", "AAPL 10-K 2024", "NVDA 10-K 2024"],
        expected_sources=["AAPL 10-K 2024"],
    )
    # AAPL is at rank 2: DCG = 1/log2(3), ideal DCG = 1/log2(2)
    # NDCG = (1/log2(3)) / (1/log2(2)) = log2(2)/log2(3) ≈ 0.6309
    expected = math.log2(2) / math.log2(3)
    assert abs(score - expected) < 1e-4


def test_ndcg_single_relevant_doc():
    from evals.metrics.retrieval import NDCGMetric
    metric = NDCGMetric(k=3)
    # Should not crash on single retrieved document
    score = metric.compute(
        retrieved_sources=["AAPL 10-K 2024"],
        expected_sources=["AAPL 10-K 2024"],
    )
    assert score == 1.0


def test_numerical_accuracy_exact():
    from evals.metrics.due_diligence import NumericalAccuracyMetric
    m = NumericalAccuracyMetric()
    score = m.compute(
        answer="Apple's FY2024 revenue was $391.0 billion.",
        expected_numbers=[391.0],
    )
    assert score == 1.0


def test_numerical_accuracy_within_tolerance():
    from evals.metrics.due_diligence import NumericalAccuracyMetric
    m = NumericalAccuracyMetric()
    # 392.0 is within 1% of 391.0 (diff = 0.256%)
    score = m.compute(
        answer="Revenue was approximately $392.0 billion.",
        expected_numbers=[391.0],
    )
    assert score == 1.0


def test_numerical_accuracy_wrong():
    from evals.metrics.due_diligence import NumericalAccuracyMetric
    m = NumericalAccuracyMetric()
    score = m.compute(
        answer="Revenue was $300 billion.",
        expected_numbers=[391.0],
    )
    assert score == 0.0


def test_entity_match_all_present():
    from evals.metrics.due_diligence import EntityMatchPrecisionMetric
    m = EntityMatchPrecisionMetric()
    score = m.compute(
        answer="Apple reported FY2024 revenue of $391.0 billion.",
        expected_entities=["Apple", "FY2024", "$391.0 billion"],
    )
    assert score == 1.0


def test_entity_match_partial():
    from evals.metrics.due_diligence import EntityMatchPrecisionMetric
    m = EntityMatchPrecisionMetric()
    score = m.compute(
        answer="Apple reported revenue.",
        expected_entities=["Apple", "FY2024", "$391.0 billion"],
    )
    assert abs(score - 1/3) < 0.01


def test_citation_accuracy_full_overlap():
    from evals.metrics.due_diligence import CitationAccuracyMetric
    m = CitationAccuracyMetric()
    score = m.compute(
        cited_sources=["AAPL 10-K 2024", "AAPL XBRL FY2024"],
        expected_sources=["AAPL 10-K 2024", "AAPL XBRL FY2024"],
    )
    assert score == 1.0


def test_citation_accuracy_no_overlap():
    from evals.metrics.due_diligence import CitationAccuracyMetric
    m = CitationAccuracyMetric()
    score = m.compute(
        cited_sources=["MSFT 10-K 2024"],
        expected_sources=["AAPL 10-K 2024"],
    )
    assert score == 0.0


def test_extract_numbers_ignores_years():
    from evals.metrics.due_diligence import _extract_numbers
    result = _extract_numbers("Apple FY2024 revenue was $391.0 billion.")
    assert 2024.0 not in result
    assert 391.0 in result


def test_completeness_metric_mocked():
    from unittest.mock import MagicMock
    from evals.metrics.due_diligence import CompletenessMetric

    mock_judge = MagicMock()
    mock_judge.generate.return_value = '{"score": 0.9, "reason": "covers all key points"}'

    m = CompletenessMetric(model=mock_judge)
    score = m.compute(
        actual_output="Apple FY2024 revenue was $391.0B driven by iPhone sales.",
        expected_output="Apple's total revenue for FY2024 was $391.0 billion.",
    )
    assert score == 0.9


def test_due_diligence_confidence_weighted():
    from evals.metrics.due_diligence import DueDiligenceConfidenceScore
    score = DueDiligenceConfidenceScore.weighted_score(
        faithfulness=1.0,
        completeness=1.0,
        numerical_accuracy=1.0,
        citation_accuracy=1.0,
    )
    assert abs(score - 1.0) < 0.001


def test_due_diligence_confidence_partial():
    from evals.metrics.due_diligence import DueDiligenceConfidenceScore
    score = DueDiligenceConfidenceScore.weighted_score(
        faithfulness=0.0,
        completeness=0.0,
        numerical_accuracy=0.0,
        citation_accuracy=0.0,
    )
    assert score == 0.0


def test_contradiction_detection_rate_empty_contradictions():
    from evals.metrics.due_diligence import ContradictionDetectionRate
    m = ContradictionDetectionRate()
    # Empty expected_contradictions → vacuously 1.0
    score = m.compute(actual_output="Some answer.", expected_contradictions=[])
    assert score == 1.0


def test_contradiction_detection_rate_mocked():
    from unittest.mock import MagicMock
    from evals.metrics.due_diligence import ContradictionDetectionRate

    mock_judge = MagicMock()
    mock_judge.generate.return_value = '{"score": 0.75, "reason": "found 3 of 4 contradictions"}'

    m = ContradictionDetectionRate(model=mock_judge)
    score = m.compute(
        actual_output="Nvidia's Q4 guidance was lower than actuals.",
        expected_contradictions=["Revenue guidance overstated by 15%"],
    )
    assert score == 0.75
