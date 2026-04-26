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
