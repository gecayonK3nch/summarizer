from summarizer_bot.models import ChatMessage
from summarizer_bot.summarizer import Summarizer


class DummyLLM:
    def __init__(self) -> None:
        self.prompt = None

    async def summarize(self, prompt: str, temperature: float = 0.2) -> str:
        self.prompt = prompt
        return "summary"


def test_build_prompt_limits_and_formats_messages() -> None:
    summarizer = Summarizer(DummyLLM(), max_chars=80)
    prompt = summarizer.build_prompt(
        [
            ChatMessage(author="Alice", text="First message"),
            ChatMessage(author="Bob", text="Second message"),
        ]
    )

    assert "Alice: First message" in prompt
    assert "Bob: Second message" in prompt
    assert "Return a concise summary" in prompt
