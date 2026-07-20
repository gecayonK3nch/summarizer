from __future__ import annotations

import re
import logging
from typing import TypedDict

from openai import APIConnectionError, APIError, AsyncOpenAI, NotFoundError, RateLimitError

logger = logging.getLogger(__name__)


class QueryAnalysis(TypedDict):
    need_search: bool
    search_query: str | None
    need_history: bool

_HISTORY_PATTERNS = (
    re.compile(
        r"\b(что мы|что я|что ты|что там|о чем мы|о чём мы|о чем говорили|о чём говорили|"
        r"напомни|вспомни|ранее|выше|ниже|как там|что было|что говорили|помнишь|"
        r"продолжим|на чем остановились|на чём остановились|в прошлый раз|по этому поводу)\b",
        re.IGNORECASE,
    ),
)

_SEARCH_PATTERNS = (
    re.compile(
        r"\b(сейчас|сегодня|сегодняшн\w*|текущ\w*|актуаль\w*|последн\w*|"
        r"обновлен\w*|новост\w*|курс\w*|цена\w*|стоимост\w*|погода|прогноз|"
        r"релиз\w*|верси\w*|доступн\w*|работает ли|кто сейчас|сколько стоит|"
        r"что нового|изменилось ли|проверь|посмотри|найди|поиск|интернет|в сети|в интернете|web|online)\b",
        re.IGNORECASE,
    ),
)


def _classify_query(prompt: str) -> QueryAnalysis:
    text = prompt.strip()
    lowered = text.casefold()

    need_history = any(pattern.search(lowered) for pattern in _HISTORY_PATTERNS)
    need_search = any(pattern.search(lowered) for pattern in _SEARCH_PATTERNS)

    return {
        "need_search": need_search,
        "search_query": text if need_search else None,
        "need_history": need_history,
    }


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
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=30.0)

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
                            "content": (
                                "You summarize Telegram conversations clearly, briefly, and accurately. "
                                "Format the summary in standard Markdown (bullet lists, **bold** for key points); "
                                "Telegram renders it natively."
                            ),
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

    async def analyze_query(self, prompt: str) -> QueryAnalysis:
        """Classify a question locally to avoid a second LLM round-trip."""
        return _classify_query(prompt)

    async def answer_question(
        self, prompt: str, history: str | None = None, search_results: str | None = None
    ) -> str:
        """Answers the user's question using optional context."""
        system_content = (
            "You are a helpful and intelligent Telegram bot. "
            "IMPORTANT: Answer clearly, directly, and concisely. DO NOT output long, verbose, or exhaustive encyclopedic text. "
            "Avoid information noise; provide only the essential facts. "
            "Format your reply in standard Markdown, which Telegram renders natively. You may use: "
            "**bold**, *italic*, ~~strikethrough~~, `inline code`, bullet and numbered lists, > blockquotes, "
            "# headings, GitHub-style tables, and fenced code blocks with a language tag (e.g. ```python\\ncode\\n```). "
            "For mathematics, use LaTeX: inline math as $...$ and display formulas as $$...$$. "
            "Apply formatting only where it genuinely improves readability; for a short answer, plain sentences are best — "
            "do not add headings or tables to a one-line reply. "
            "If you use search results, briefly list the sources at the bottom."
        )
        if history:
            system_content += f"\n\nRecent chat history for context:\n{history}\n"
        if search_results:
            system_content += f"\n\nSearch results from the web to help answer the question:\n{search_results}\n"

        last_error: Exception | None = None
        for index, model in enumerate(self._models):
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    temperature=0.6,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = (response.choices[0].message.content or "").strip()
                if content:
                    return content
                last_error = RuntimeError(f"Model '{model}' returned empty content")
            except Exception as e:
                last_error = e
                continue
        raise LLMUnavailableError("No available models could answer the question") from last_error
