from summarizer_bot.llm import _classify_query


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