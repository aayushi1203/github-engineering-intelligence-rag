from typing import Protocol

import httpx

from github_intelligence.generation.models import DraftAnswer


class LLMProvider(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str) -> DraftAnswer: ...


class LLMProviderError(RuntimeError):
    pass


class OpenAIResponsesProvider:
    """OpenAI Responses API adapter using strict JSON-schema output."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        api_url: str = "https://api.openai.com/v1",
        max_output_tokens: int = 1_200,
        timeout: float = 60,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for answer generation")
        self.api_key = api_key
        self.model = model
        self.api_url = api_url.rstrip("/")
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    def generate(self, *, system_prompt: str, user_prompt: str) -> DraftAnswer:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_output_tokens": self.max_output_tokens,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "grounded_github_answer",
                    "strict": True,
                    "schema": DraftAnswer.model_json_schema(),
                }
            },
        }
        try:
            response = httpx.post(
                f"{self.api_url}/responses",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout,
            )
            if response.is_error:
                raise LLMProviderError(
                    f"OpenAI API returned {response.status_code}: {response.text}"
                )
            data = response.json()
            output_text = data.get("output_text") or self._find_output_text(data)
            if not output_text:
                raise LLMProviderError("OpenAI response contained no output text")
            return DraftAnswer.model_validate_json(output_text)
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMProviderError(f"OpenAI generation failed: {exc}") from exc

    @staticmethod
    def _find_output_text(data: dict[str, object]) -> str | None:
        output = data.get("output")
        if not isinstance(output, list):
            return None
        for item in output:
            if not isinstance(item, dict) or not isinstance(item.get("content"), list):
                continue
            for content in item["content"]:
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        return text
        return None
