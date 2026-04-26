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
