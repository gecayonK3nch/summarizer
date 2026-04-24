from __future__ import annotations

import logging

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message

from .models import ChatMessage
from .storage import MessageStore
from .summarizer import Summarizer
from .config import Settings
from .llm import LLMUnavailableError

router = Router()
logger = logging.getLogger(__name__)


def build_router(store: MessageStore, summarizer: Summarizer, settings: Settings) -> Router:
    local_router = Router()

    @local_router.message(Command("start"))
    async def start_handler(message: Message) -> None:
        await message.answer(
            "Send messages in the chat, then use /summary N to summarize the last N messages."
        )

    @local_router.message(Command("help"))
    async def help_handler(message: Message) -> None:
        await message.answer(
            f"Use /summary N, where N is between 1 and {settings.summary_max_messages}."
        )

    @local_router.message(Command("summary"))
    async def summary_handler(message: Message) -> None:
        parts = (message.text or "").split(maxsplit=1)
        requested = 10
        if len(parts) > 1:
            try:
                requested = int(parts[1])
            except ValueError:
                await message.answer("Use /summary N with a positive integer.")
                return

        if requested < 1:
            await message.answer("N must be at least 1.")
            return

        requested = min(requested, settings.summary_max_messages)
        recent_messages = store.get_recent_messages(message.chat.id, requested)
        if not recent_messages:
            await message.answer("I do not have enough stored messages yet.")
            return

        try:
            summary = await summarizer.summarize_messages(
                recent_messages, temperature=settings.summary_temperature
            )
        except LLMUnavailableError:
            await message.answer(
                "All configured free models are unavailable right now. Try again soon or change OPENAI_MODEL."
            )
            return
        except Exception:
            logger.exception("Summary request failed for chat %s", message.chat.id)
            await message.answer("Could not generate summary due to provider error.")
            return

        await message.answer(summary)

    @local_router.message(F.text & ~F.text.startswith("/"))
    async def store_message(message: Message) -> None:
        author = message.from_user.full_name if message.from_user else "Unknown"
        store.add_message(message.chat.id, ChatMessage(author=author, text=message.text or ""))
        logger.debug("Stored message in chat %s", message.chat.id)

    return local_router
