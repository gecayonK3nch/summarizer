from __future__ import annotations

from .models import ChatMessage
from .llm import LLMClient


class Summarizer:
    def __init__(self, llm_client: LLMClient, max_chars: int = 20_000) -> None:
        self._llm_client = llm_client
        self._max_chars = max_chars

    def build_prompt(self, messages: list[ChatMessage]) -> str:
        lines = ["Summarize the conversation below in Russian."]
        lines.append("Focus on key decisions, questions, open threads, and action items.")
        lines.append("")
        lines.append("Messages:")

        total_chars = 0
        for message in messages:
            line = f"- {message.author}: {message.text}"
            if total_chars + len(line) > self._max_chars:
                break
            lines.append(line)
            total_chars += len(line)

        lines.append("")
        lines.append("Return a concise summary with up to 5 bullets and no preamble.")
        return "\n".join(lines)

    async def summarize_messages(self, messages: list[ChatMessage], temperature: float = 0.2) -> str:
        prompt = self.build_prompt(messages)
        return await self._llm_client.summarize(prompt, temperature=temperature)
