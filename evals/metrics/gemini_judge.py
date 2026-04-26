from __future__ import annotations

from deepeval.models.base_model import DeepEvalBaseLLM

from src.model_config import get_genai_client, get_model_name


class GeminiJudge(DeepEvalBaseLLM):
    def get_model_name(self) -> str:
        return get_model_name()

    def load_model(self):
        return get_genai_client()

    def generate(self, prompt: str) -> str:
        client = get_genai_client()
        response = client.models.generate_content(
            model=get_model_name(),
            contents=prompt,
        )
        return response.text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
