from __future__ import annotations

from collections import defaultdict, deque

from .models import ChatMessage


class MessageStore:
    def __init__(self, max_messages_per_chat: int = 200) -> None:
        self._max_messages_per_chat = max_messages_per_chat
        self._messages: dict[int, deque[ChatMessage]] = defaultdict(
            lambda: deque(maxlen=self._max_messages_per_chat)
        )

    def add_message(self, chat_id: int, message: ChatMessage) -> None:
        self._messages[chat_id].append(message)

    def get_recent_messages(self, chat_id: int, count: int) -> list[ChatMessage]:
        if count <= 0:
            return []
        return list(self._messages.get(chat_id, deque()))[-count:]
