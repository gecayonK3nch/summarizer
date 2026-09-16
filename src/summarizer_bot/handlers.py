from __future__ import annotations

import asyncio
import logging
import re

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import InputRichMessage, Message
from aiogram.utils.chat_action import ChatActionSender
from ddgs import DDGS

from .config import Settings
from .llm import LLMUnavailableError
from .models import ChatMessage
from .storage import MessageStore
from .summarizer import Summarizer

logger = logging.getLogger(__name__)

# Rich messages accept up to 32768 UTF-8 characters (Bot API 10.1+). Guard
# against pathological over-long model output before attempting a rich send.
RICH_MESSAGE_LIMIT = 32_768

# Plain-text messages are capped at 4096 characters; keep some headroom.
PLAIN_CHUNK_SIZE = 4000

DEFAULT_SUMMARY_SIZE = 10

# Telegram rich Markdown recognizes math only as $...$ / $$...$$ (and ```math```).
# LLMs frequently emit the \(...\) and \[...\] delimiters instead, which Telegram
# then shows as raw text. Rewrite them to the dollar form before sending.
# Split on code spans first so we never touch backslashes inside real code
# (e.g. a Python regex like r"\(").
_CODE_SPAN_RE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_DISPLAY_MATH_DELIM_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH_DELIM_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def _normalize_math_delimiters(text: str) -> str:
    """Convert ``\\[...\\]`` -> ``$$...$$`` and ``\\(...\\)`` -> ``$...$``.

    Only applied outside fenced/inline code, so code containing escaped parens or
    brackets is left untouched.
    """
    def convert(segment: str) -> str:
        segment = _DISPLAY_MATH_DELIM_RE.sub(lambda m: f"$${m.group(1)}$$", segment)
        segment = _INLINE_MATH_DELIM_RE.sub(lambda m: f"${m.group(1)}$", segment)
        return segment

    out: list[str] = []
    last = 0
    for match in _CODE_SPAN_RE.finditer(text):
        out.append(convert(text[last:match.start()]))
        out.append(match.group(0))  # code span kept verbatim
        last = match.end()
    out.append(convert(text[last:]))
    return "".join(out)


async def _perform_web_search(query: str) -> str:
    def sync_search() -> str:
        try:
            results = list(DDGS().text(query, max_results=3))
        except Exception:
            logger.exception("DuckDuckGo search failed for query %r", query)
            return "Search failed."
        if not results:
            return "No useful search results found."
        return "\n\n".join(
            f"Title: {r.get('title')}\nSnippet: {r.get('body')}" for r in results
        )

    return await asyncio.to_thread(sync_search)


async def _send_rich(message: Message, markdown: str) -> None:
    """Send the model's Markdown reply as a native Telegram rich message.

    Rich messages (Bot API 10.1+) render standard Markdown natively: headings,
    tables, lists, blockquotes, fenced code blocks and LaTeX math (``$...$`` /
    ``$$...$$``). We therefore forward the model output verbatim instead of the
    old HTML/PNG post-processing. If the rich send fails (malformed markup or an
    oversized payload), fall back to a plain-text reply so the user still gets an
    answer.
    """
    text = _normalize_math_delimiters((markdown or "").strip())
    if not text:
        return

    if len(text) <= RICH_MESSAGE_LIMIT:
        try:
            await message.reply_rich(rich_message=InputRichMessage(markdown=text))
            return
        except Exception:
            logger.exception("Rich message send failed; falling back to plain text")

    await _send_plain(message, text)


async def _send_plain(message: Message, text: str) -> None:
    """Fallback sender: split long text into plain-text chunks under Telegram's limit."""
    for i in range(0, len(text), PLAIN_CHUNK_SIZE):
        await message.reply(text[i:i + PLAIN_CHUNK_SIZE], parse_mode=None)


def build_router(store: MessageStore, summarizer: Summarizer, settings: Settings) -> Router:
    router = Router()

    @router.message(Command("start"))
    async def start_handler(message: Message) -> None:
        await message.reply(
            "Send messages in the chat, then use /summary N to summarize the last N messages.\n"
            "Use /ask <question> or !ask <question> to ask me anything."
        )

    @router.message(Command("help"))
    async def help_handler(message: Message) -> None:
        await message.reply(
            f"Use /summary N, where N is between 1 and {settings.summary_max_messages} "
            f"(default {DEFAULT_SUMMARY_SIZE}).\n"
            "Use /ask <question> or !ask <question> to ask a question with optional internet search."
        )

    @router.message(Command("summary"))
    async def summary_handler(message: Message, command: CommandObject) -> None:
        requested = DEFAULT_SUMMARY_SIZE
        if command.args:
            try:
                requested = int(command.args)
            except ValueError:
                await message.reply("Use /summary N with a positive integer.")
                return

        if requested < 1:
            await message.reply("N must be at least 1.")
            return

        requested = min(requested, settings.summary_max_messages)
        recent_messages = store.get_recent_messages(message.chat.id, requested)
        if not recent_messages:
            await message.reply("I do not have enough stored messages yet.")
            return

        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            try:
                summary = await summarizer.summarize_messages(
                    recent_messages, temperature=settings.summary_temperature
                )
            except LLMUnavailableError:
                await message.reply(
                    "All configured free models are unavailable right now. Try again soon or change OPENAI_MODEL."
                )
                return
            except Exception:
                logger.exception("Summary request failed for chat %s", message.chat.id)
                await message.reply("Could not generate summary due to provider error.")
                return

            await _send_rich(message, summary)

    @router.message(Command("ask"))
    @router.message(F.text.startswith("!ask "))
    async def ask_handler(message: Message, command: CommandObject | None = None) -> None:
        # `command` is injected by the Command filter only; the "!ask" route has none.
        if command is not None:
            question = (command.args or "").strip()
        else:
            question = (message.text or "")[len("!ask "):].strip()

        if not question:
            await message.reply("Please provide a question after the command.")
            return

        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            try:
                # 1. Decide locally which extra context the question needs.
                analysis = await summarizer.analyze_query(question)

                history_text = None
                if analysis["need_history"]:
                    recent_msgs = store.get_recent_messages(message.chat.id, settings.summary_max_messages)
                    if recent_msgs:
                        history_text = "\n".join(f"{m.author}: {m.text}" for m in recent_msgs)

                search_results = None
                if analysis["need_search"] and analysis["search_query"]:
                    search_results = await _perform_web_search(analysis["search_query"])

                # 2. Get the final answer.
                answer = await summarizer.answer_question(
                    prompt=question,
                    history=history_text,
                    search_results=search_results,
                )
                await _send_rich(message, answer)
            except LLMUnavailableError:
                await message.reply("AI models are currently unavailable. Please try again later.")
            except Exception:
                logger.exception("Ask request failed for chat %s", message.chat.id)
                await message.reply("I had trouble processing that question.")

    @router.message(F.text & ~F.text.startswith("/"))
    async def store_message(message: Message) -> None:
        author = message.from_user.full_name if message.from_user else "Unknown"
        store.add_message(message.chat.id, ChatMessage(author=author, text=message.text or ""))
        logger.debug("Stored message in chat %s", message.chat.id)

    return router
