from __future__ import annotations

import asyncio
import logging

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender
from ddgs import DDGS

from .models import ChatMessage
from .storage import MessageStore
from .summarizer import Summarizer
from .config import Settings
from .llm import LLMUnavailableError

router = Router()
logger = logging.getLogger(__name__)

async def _perform_web_search(query: str) -> str:
    def sync_search() -> str:
        try:
            results = list(DDGS().text(query, max_results=3))
            if not results:
                return "No useful search results found."
            chunks = []
            for r in results:
                chunks.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}")
            return "\n\n".join(chunks)
        except Exception as e:
            logger.error("DuckDuckGo search error: %s", e)
            return "Search failed."

    return await asyncio.to_thread(sync_search)

async def _send_long_message(message: Message, text: str, parse_mode: str | None = None) -> None:
    """Helper to send potentially long messages by splitting them into chunks."""
    max_chunk_size = 4000
    if not text:
        return
    for i in range(0, len(text), max_chunk_size):
        chunk = text[i:i + max_chunk_size]
        try:
            await message.answer(chunk, parse_mode=parse_mode)
        except Exception:
            # Fallback if parsing fails (e.g. unclosed HTML tags)
            await message.answer(chunk)

def build_router(store: MessageStore, summarizer: Summarizer, settings: Settings) -> Router:
    local_router = Router()

    @local_router.message(Command("start"))
    async def start_handler(message: Message) -> None:
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            await message.answer(
                "Send messages in the chat, then use /summary N to summarize the last N messages.\n"
                "Use /ask <question> or !ask <question> to ask me anything."
            )

    @local_router.message(Command("help"))
    async def help_handler(message: Message) -> None:
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            await message.answer(
                f"Use /summary N, where N is between 1 and {settings.summary_max_messages}.\n"
                "Use /ask <question> or !ask <question> to ask a question with optional internet search."
            )

    @local_router.message(Command("summary"))
    async def summary_handler(message: Message) -> None:
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
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

            await _send_long_message(message, summary)

    @local_router.message(Command("ask"))
    @local_router.message(F.text.startswith("!ask "))
    async def ask_handler(message: Message) -> None:
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            if message.text.startswith("!ask "):
                question = message.text[5:].strip()
            else:
                parts = (message.text or "").split(maxsplit=1)
                question = parts[1].strip() if len(parts) > 1 else ""

            if not question:
                await message.answer("Please provide a question after the command.")
                return

            try:
                # 1. Analyze the query to see what context is needed
                analysis = await summarizer.analyze_query(question)
                
                history_text = None
                if analysis.get("need_history"):
                    recent_msgs = store.get_recent_messages(message.chat.id, settings.summary_max_messages)
                    if recent_msgs:
                        history_text = "\n".join(f"{m.author}: {m.text}" for m in recent_msgs)
                
                search_results = None
                if analysis.get("need_search") and analysis.get("search_query"):
                    search_results = await _perform_web_search(analysis["search_query"])
                
                # 2. Get the final answer
                answer = await summarizer.answer_question(
                    prompt=question,
                    history=history_text,
                    search_results=search_results
                )
                await _send_long_message(message, answer, parse_mode="HTML")
            except LLMUnavailableError:
                await message.answer("AI models are currently unavailable. Please try again later.")
            except Exception:
                logger.exception("Ask request failed for chat %s", message.chat.id)
                await message.answer("I had trouble processing that question.")

    @local_router.message(F.text & ~F.text.startswith("/"))
    async def store_message(message: Message) -> None:
        author = message.from_user.full_name if message.from_user else "Unknown"
        store.add_message(message.chat.id, ChatMessage(author=author, text=message.text or ""))
        logger.debug("Stored message in chat %s", message.chat.id)

    return local_router
