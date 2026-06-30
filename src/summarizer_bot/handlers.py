from __future__ import annotations

import asyncio
import logging
import io
import re
import html

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, InputFile
from aiogram.utils.chat_action import ChatActionSender
from ddgs import DDGS

# Matplotlib for rendering LaTeX to PNG
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pygments.lexers import get_lexer_by_name

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


# Matplotlib mathtext only understands a subset of LaTeX. These commands have
# no mathtext equivalent and must be stripped before rendering.
_MATHTEXT_STRIP_RE = re.compile(r"\\(?:bigg?|Bigg?)[lrm]?\b|\\(?:left|right|displaystyle|limits)\b")


def _sanitize_latex_for_mathtext(latex: str) -> str:
    """Make a LaTeX snippet renderable by matplotlib's mathtext engine."""
    cleaned = _MATHTEXT_STRIP_RE.sub("", latex)
    # mathtext uses \mathbf{...}; bare "\mathbf r" style needs braces.
    cleaned = re.sub(r"\\(mathbf|mathrm|mathbb|mathcal|boldsymbol|mathit)\s+([A-Za-z])", r"\\\1{\2}", cleaned)
    return cleaned.strip()


def _render_latex_to_png_bytes(latex: str) -> io.BytesIO:
    """Render LaTeX math (display style) to a PNG and return BytesIO."""
    safe = _sanitize_latex_for_mathtext(latex)
    fig = plt.figure()
    fig.text(0, 0, f"${safe}$", fontsize=18)
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=200, transparent=False, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


async def _send_image_bytes(message: Message, image_buf: io.BytesIO, caption: str | None = None) -> None:
    image_buf.seek(0)
    # send as a reply-photo to the triggering message
    # Pass the BytesIO directly to InputFile without duplicate filename kwarg.
    await message.reply_photo(photo=InputFile(image_buf), caption=caption)


# --- Inline LaTeX -> readable Unicode -------------------------------------

_LATEX_SYMBOLS = {
    r"\int": "∫", r"\oint": "∮", r"\sum": "∑", r"\prod": "∏",
    r"\partial": "∂", r"\nabla": "∇", r"\infty": "∞",
    r"\cdot": "·", r"\times": "×", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈", r"\equiv": "≡",
    r"\rightarrow": "→", r"\to": "→", r"\leftarrow": "←", r"\Rightarrow": "⇒",
    r"\in": "∈", r"\notin": "∉", r"\subset": "⊂", r"\subseteq": "⊆",
    r"\forall": "∀", r"\exists": "∃", r"\angle": "∠", r"\propto": "∝",
    r"\langle": "⟨", r"\rangle": "⟩", r"\dots": "…", r"\ldots": "…", r"\cdots": "⋯",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ε",
    r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η", r"\theta": "θ", r"\kappa": "κ",
    r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ", r"\pi": "π", r"\rho": "ρ",
    r"\sigma": "σ", r"\tau": "τ", r"\phi": "φ", r"\varphi": "φ", r"\chi": "χ",
    r"\psi": "ψ", r"\omega": "ω", r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ",
    r"\Lambda": "Λ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
}

_SUB_MAP = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
    "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ", "l": "ₗ", "m": "ₘ",
    "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ",
}

_SUP_MAP = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "n": "ⁿ", "i": "ⁱ",
}


def _to_script(content: str, table: dict[str, str], marker: str) -> str:
    if content and all(ch in table for ch in content):
        return "".join(table[ch] for ch in content)
    return f"{marker}{content}" if len(content) == 1 else f"{marker}({content})"


def _latex_to_unicode(snippet: str) -> str:
    """Convert an inline LaTeX math snippet to a readable plain-text approximation."""
    s = snippet.strip()
    # Font/style wrappers: keep the content only.
    s = re.sub(r"\\(?:mathbf|mathrm|mathbb|mathcal|boldsymbol|mathit|vec|hat|bar|text|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathbf|mathrm|mathbb|mathcal|boldsymbol|mathit|vec|hat|bar)\s+([A-Za-z])", r"\1", s)
    # Fractions and roots.
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)", s)
    # Norms and absolute values.
    s = s.replace(r"\|", "‖")
    # Size/spacing delimiters that have no symbol meaning.
    s = _MATHTEXT_STRIP_RE.sub("", s)
    # Named symbols and Greek letters (sort by length to avoid partial clashes).
    for cmd in sorted(_LATEX_SYMBOLS, key=len, reverse=True):
        s = s.replace(cmd, _LATEX_SYMBOLS[cmd])
    # Primes.
    s = s.replace("'", "′")
    # Sub/superscripts.
    s = re.sub(r"_\{([^{}]*)\}", lambda m: _to_script(m.group(1), _SUB_MAP, "_"), s)
    s = re.sub(r"_([0-9A-Za-z])", lambda m: _to_script(m.group(1), _SUB_MAP, "_"), s)
    s = re.sub(r"\^\{([^{}]*)\}", lambda m: _to_script(m.group(1), _SUP_MAP, "^"), s)
    s = re.sub(r"\^([0-9A-Za-z])", lambda m: _to_script(m.group(1), _SUP_MAP, "^"), s)
    # Thin spaces and any leftover commands.
    s = re.sub(r"\\[,;:!> ]", " ", s)
    s = re.sub(r"\\[A-Za-z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"[ \t]+", " ", s).strip()


def _code_block_to_html(lang: str | None, code: str) -> str:
    """Build a Telegram-safe <pre><code> block, tagging the language when known."""
    escaped = html.escape(code.strip("\n"))
    if lang:
        try:
            get_lexer_by_name(lang, stripall=True)
            return f'<pre><code class="language-{html.escape(lang)}">{escaped}</code></pre>'
        except Exception:
            pass
    return f"<pre><code>{escaped}</code></pre>"


_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]+)?[ \t]*\n?(.*?)```", re.DOTALL)
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)|(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)


def _render_inline_text(text: str) -> str:
    """Convert inline math in a text run to Unicode, preserving surrounding model HTML."""
    def repl(match: re.Match) -> str:
        inner = match.group(1) if match.group(1) is not None else match.group(2)
        return html.escape(_latex_to_unicode(inner))

    return _INLINE_MATH_RE.sub(repl, text)


async def _process_and_send(message: Message, text: str, parse_mode: str | None = "HTML") -> None:
    """Send a message, rendering display math as images and inline math/code inline.

    - ```code``` fences become Telegram <pre><code> blocks.
    - Display math (``$$...$$`` or ``\\[...\\]``) is rendered to a PNG image.
    - Inline math (``\\(...\\)`` or ``$...$``) is converted to readable Unicode text.
    """
    # 1. Stash code blocks so math processing never touches their contents.
    code_blocks: list[str] = []

    def _stash(match: re.Match) -> str:
        code_blocks.append(_code_block_to_html(match.group(1), match.group(2)))
        return f"\x00CODE{len(code_blocks) - 1}\x00"

    staged = _FENCE_RE.sub(_stash, text)

    def _restore(chunk: str) -> str:
        return re.sub(r"\x00CODE(\d+)\x00", lambda m: code_blocks[int(m.group(1))], chunk)

    # 2. Walk the text, splitting on display-math blocks.
    pos = 0
    for match in _DISPLAY_MATH_RE.finditer(staged):
        preceding = staged[pos:match.start()]
        if preceding.strip():
            await _send_long_message(message, _restore(_render_inline_text(preceding)), parse_mode=parse_mode)
        formula = (match.group(1) if match.group(1) is not None else match.group(2)).strip()
        try:
            img_buf = _render_latex_to_png_bytes(formula)
            await _send_image_bytes(message, img_buf)
        except Exception as error:
            logger.warning("Failed to render LaTeX '%s': %s", formula, error)
            await _send_long_message(message, _latex_to_unicode(formula), parse_mode=None)
        pos = match.end()

    trailing = staged[pos:]
    if trailing.strip():
        await _send_long_message(message, _restore(_render_inline_text(trailing)), parse_mode=parse_mode)

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

            await _process_and_send(message, summary)

    @local_router.message(Command("ask"))
    @local_router.message(F.text.startswith("!ask "))
    async def ask_handler(message: Message) -> None:
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            text = message.text or ""
            if text.startswith("!ask "):
                question = text[5:].strip()
            else:
                parts = text.split(maxsplit=1)
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
                search_query = analysis.get("search_query")
                if analysis.get("need_search") and isinstance(search_query, str):
                    search_results = await _perform_web_search(search_query)
                
                # 2. Get the final answer
                answer = await summarizer.answer_question(
                    prompt=question,
                    history=history_text,
                    search_results=search_results
                )
                await _process_and_send(message, answer, parse_mode="HTML")
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
