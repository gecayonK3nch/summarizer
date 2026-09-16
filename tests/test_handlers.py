from summarizer_bot.handlers import _normalize_math_delimiters


def test_normalize_converts_latex_delimiters_to_dollars() -> None:
    text = r"Inline \(a^2 + b^2\) and display \[E = mc^2\] math."

    assert _normalize_math_delimiters(text) == "Inline $a^2 + b^2$ and display $$E = mc^2$$ math."


def test_normalize_leaves_code_spans_untouched() -> None:
    text = 'Regex `r"\\(foo\\)"` and\n```python\nprint("\\[x\\]")\n```\nbut \\(y\\) outside.'

    result = _normalize_math_delimiters(text)

    assert '`r"\\(foo\\)"`' in result
    assert 'print("\\[x\\]")' in result
    assert result.endswith("but $y$ outside.")


def test_normalize_keeps_dollar_math_as_is() -> None:
    text = "Already $x$ and $$y$$."

    assert _normalize_math_delimiters(text) == text
