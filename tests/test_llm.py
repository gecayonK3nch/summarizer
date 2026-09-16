from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, NotFoundError

from summarizer_bot.llm import LLMClient, LLMUnavailableError, _classify_query


def test_classify_query_uses_history_for_followup_questions() -> None:
    result = _classify_query("Напомни, что мы обсуждали выше?")

    assert result == {
        "need_search": False,
        "search_query": None,
        "need_history": True,
    }


def test_classify_query_uses_search_for_current_information() -> None:
    result = _classify_query("Какая сейчас погода в Москве?")

    assert result == {
        "need_search": True,
        "search_query": "Какая сейчас погода в Москве?",
        "need_history": False,
    }


def test_classify_query_skips_extra_context_for_general_questions() -> None:
    result = _classify_query("Что такое санбиллютен?")

    assert result == {
        "need_search": False,
        "search_query": None,
        "need_history": False,
    }


class FakeCompletions:
    """Stand-in for ``client.chat.completions`` that replays scripted outcomes per model."""

    def __init__(self, outcomes: dict[str, object]) -> None:
        self._outcomes = outcomes
        self.calls: list[str] = []

    async def create(self, *, model: str, **_: object) -> object:
        self.calls.append(model)
        outcome = self._outcomes[model]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))])


def _make_client(outcomes: dict[str, object]) -> tuple[LLMClient, FakeCompletions]:
    client = LLMClient(api_key="k", base_url="http://localhost", model="a", fallback_models=["b", "c"])
    fake = FakeCompletions(outcomes)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))  # type: ignore[assignment]
    return client, fake


def _not_found() -> NotFoundError:
    request = httpx.Request("POST", "http://localhost")
    response = httpx.Response(404, request=request)
    return NotFoundError("missing", response=response, body=None)


def test_llm_client_dedupes_and_orders_models() -> None:
    client = LLMClient(api_key="k", base_url="http://localhost", model=" a ", fallback_models=["b", "a", "", "c"])

    assert client._models == ["a", "b", "c"]


def test_llm_client_requires_at_least_one_model() -> None:
    with pytest.raises(ValueError):
        LLMClient(api_key="k", base_url="http://localhost", model="  ", fallback_models=[""])


async def test_summarize_falls_back_past_missing_and_empty_models() -> None:
    client, fake = _make_client({"a": _not_found(), "b": "   ", "c": "ok"})

    assert await client.summarize("prompt") == "ok"
    assert fake.calls == ["a", "b", "c"]


async def test_answer_question_raises_when_every_model_fails() -> None:
    connection_error = APIConnectionError(request=httpx.Request("POST", "http://localhost"))
    client, fake = _make_client({"a": _not_found(), "b": connection_error, "c": ""})

    with pytest.raises(LLMUnavailableError):
        await client.answer_question("question", history="h", search_results="s")
    assert fake.calls == ["a", "b", "c"]
