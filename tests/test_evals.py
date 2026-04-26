from unittest.mock import MagicMock, patch


def test_gemini_judge_get_model_name():
    with patch("src.model_config.get_genai_client", return_value=MagicMock()):
        import importlib

        import evals.metrics.gemini_judge as m

        importlib.reload(m)
        judge = m.GeminiJudge()
        assert judge.get_model_name() == "gemini-2.5-flash"


def test_gemini_judge_generate_calls_genai():
    mock_response = MagicMock()
    mock_response.text = "yes"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("src.model_config.get_genai_client", return_value=mock_client):
        import importlib

        import evals.metrics.gemini_judge as m

        importlib.reload(m)
        judge = m.GeminiJudge()
        result = judge.generate("Is this faithful?")

        assert result == "yes"
        mock_client.models.generate_content.assert_called_once()
