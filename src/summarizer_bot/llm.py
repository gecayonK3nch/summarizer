from __future__ import annotations

import logging

from openai import APIConnectionError, APIError, AsyncOpenAI, NotFoundError, RateLimitError

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when all configured models fail."""


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        fallback_models: list[str] | None = None,
    ) -> None:
        candidate_models = [model, *(fallback_models or [])]
        self._models = list(dict.fromkeys(item.strip() for item in candidate_models if item.strip()))
        if not self._models:
            raise ValueError("At least one model must be configured")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def summarize(self, prompt: str, temperature: float = 0.2) -> str:
        last_error: Exception | None = None

        for index, model in enumerate(self._models):
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": "You summarize Telegram conversations clearly, briefly, and accurately.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                content = (response.choices[0].message.content or "").strip()
                if content:
                    if index > 0:
                        logger.warning("Used fallback model '%s' for summary generation", model)
                    return content
                last_error = RuntimeError(f"Model '{model}' returned empty content")
            except NotFoundError as error:
                last_error = error
                logger.warning("Model '%s' is unavailable on provider; trying next fallback", model)
            except (APIConnectionError, RateLimitError, APIError) as error:
                last_error = error
                logger.warning("Model '%s' failed with provider error; trying next fallback", model)

        raise LLMUnavailableError("No available models could generate a summary") from last_error
