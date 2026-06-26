from __future__ import annotations

import asyncio
import logging
import io
import re
import html

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender
from ddgs import DDGS

# Matplotlib for rendering LaTeX to PNG
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter

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


def _render_latex_to_png_bytes(latex: str) -> io.BytesIO:
    """Render LaTeX math (display style) to a PNG and return BytesIO."""
    fig = plt.figure()
    fig.text(0, 0, f"${latex}$", fontsize=16)
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=200, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


async def _send_image_bytes(message: Message, image_buf: io.BytesIO, caption: str | None = None) -> None:
    image_buf.seek(0)
    # send as a reply-photo to the triggering message
    await message.reply_photo(photo=image_buf, caption=caption)


def _convert_code_fences_to_html(text: str) -> str:
    """Convert ```lang\ncode``` fences to Telegram-safe HTML <pre><code> blocks and escape content."""
    fence_re = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)

    def repl(match: re.Match) -> str:
        lang = match.group(1)
        code = match.group(2)
        # Try syntax highlighting to HTML, but Telegram accepts plain <pre>
        try:
            if lang:
                lexer = get_lexer_by_name(lang, stripall=True)
            else:
                lexer = guess_lexer(code)
            formatter = HtmlFormatter(nowrap=True)
            highlighted = highlight(code, lexer, formatter)
            # Pygments HTML may contain tags; strip them by escaping original code instead
            escaped = html.escape(code)
        except Exception:
            escaped = html.escape(code)

        return f"<pre><code>{escaped}</code></pre>"

    return fence_re.sub(repl, text)


async def _process_and_send(message: Message, text: str, parse_mode: str | None = "HTML") -> None:
    """Process text for math ($$...$$) and code fences, sending images for display-math and HTML for others.

    Behavior:
    - Replace code fences with <pre><code>escaped</code></pre>
    - For display math blocks $$...$$ send preceding text, then an image of the math, then continue
    """
    # First convert code fences to HTML-safe pre blocks
    processed = _convert_code_fences_to_html(text)

    # Split by display math $$...$$
    parts = re.split(r"(\$\$(.*?)\$\$)", processed, flags=re.DOTALL)
    # parts will include the separators; iterate
    i = 0
    while i < len(parts):
        part = parts[i]
        if part.startswith("$$") and part.endswith("$$"):
            # extract inner math
            inner = part[2:-2].strip()
            try:
                img_buf = _render_latex_to_png_bytes(inner)
                await _send_image_bytes(message, img_buf)
            except Exception as e:
                logger.exception("Failed to render LaTeX: %s", e)
                # send as plain text fallback
                await _send_long_message(message, f"$$ {inner} $$", parse_mode=None)
        else:
            if part:
                await _send_long_message(message, part, parse_mode=parse_mode)
        i += 1

async def _send_long_message(message: Message, text: str, parse_mode: str | None = None) -> None:
    """Helper to send potentially long messages by splitting them into chunks."""
    max_chunk_size = 4000
    if not text:
        return
    for i in range(0, len(text), max_chunk_size):
        chunk = text[i:i + max_chunk_size]
        try:
            await message.reply(chunk, parse_mode=parse_mode)
        except Exception:
            # Fallback if parsing fails (e.g. unclosed HTML tags)
            await message.reply(chunk)

def build_router(store: MessageStore, summarizer: Summarizer, settings: Settings) -> Router:
    local_router = Router()

    @local_router.message(Command("start"))
    async def start_handler(message: Message) -> None:
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
                await message.reply(
                    "Send messages in the chat, then use /summary N to summarize the last N messages.\n"
                    "Use /ask <question> or !ask <question> to ask me anything."
                )

    @local_router.message(Command("help"))
    async def help_handler(message: Message) -> None:
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
                await message.reply(
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
                await message.reply("Please provide a question after the command.")
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
                await message.reply("AI models are currently unavailable. Please try again later.")
            except Exception:
                logger.exception("Ask request failed for chat %s", message.chat.id)
                await message.reply("I had trouble processing that question.")

    @local_router.message(F.text & ~F.text.startswith("/"))
    async def store_message(message: Message) -> None:
        author = message.from_user.full_name if message.from_user else "Unknown"
        store.add_message(message.chat.id, ChatMessage(author=author, text=message.text or ""))
        logger.debug("Stored message in chat %s", message.chat.id)

    return local_router
