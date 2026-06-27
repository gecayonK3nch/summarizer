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

    async def analyze_query(self, prompt: str) -> dict[str, str | bool]:
        """Analyzes a question to determine if search or history is needed."""
        system_prompt = (
            "You are an AI assistant orchestrator. Your job is to decide if a user's question "
            "needs an internet search to get up-to-date or factual info, or if it needs recent chat history context.\n"
            "Reply strictly with these three lines:\n"
            "SEARCH: [Yes/No]\n"
            "SEARCH_QUERY: [query if Yes, else None]\n"
            "HISTORY: [Yes/No]\n\n"
            "Examples:\n"
            "User: What is the weather in Sevastopol today?\n"
            "SEARCH: Yes\n"
            "SEARCH_QUERY: current weather in Sevastopol\n"
            "HISTORY: No\n\n"
            "User: Write a bedtime story.\n"
            "SEARCH: No\n"
            "SEARCH_QUERY: None\n"
            "HISTORY: No\n\n"
            "User: What were we just talking about?\n"
            "SEARCH: No\n"
            "SEARCH_QUERY: None\n"
            "HISTORY: Yes"
        )
        for index, model in enumerate(self._models):
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = (response.choices[0].message.content or "").strip()
                result = {"need_search": False, "search_query": None, "need_history": False}
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith("SEARCH:"):
                        result["need_search"] = "yes" in line.upper()
                    elif line.startswith("SEARCH_QUERY:"):
                        idx = line.find(":") + 1
                        q = line[idx:].strip()
                        if q.lower() != "none" and result["need_search"]:
                            result["search_query"] = q
                    elif line.startswith("HISTORY:"):
                        result["need_history"] = "yes" in line.upper()
                return result
            except (APIConnectionError, RateLimitError, APIError, NotFoundError):
                continue
        # Default fallback
        return {"need_search": False, "search_query": None, "need_history": False}

    async def answer_question(
        self, prompt: str, history: str | None = None, search_results: str | None = None
    ) -> str:
        """Answers the user's question using optional context."""
        system_content = (
            "You are a helpful and intelligent Telegram bot. "
            "IMPORTANT: Answer clearly, directly, and concisely. DO NOT output long, verbose, or exhaustive encyclopedic text. "
            "Avoid information noise; provide only the essential facts. "
            "Format inline text using ONLY Telegram-supported HTML tags (<b>bold</b>, <i>italic</i>, <code>code</code>, a href). "
            "DO NOT use Markdown asterisks or underscores, DO NOT use markdown tables or markdown headers. "
            "For source code, ALWAYS use a fenced block with a language tag, e.g. ```python\\ncode\\n```. "
            "For mathematics, ALWAYS use LaTeX: wrap display formulas in $$...$$ and inline math in $...$. "
            "Do NOT wrap math in <code> tags and do NOT use \\[ \\] or \\( \\) delimiters. "
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
