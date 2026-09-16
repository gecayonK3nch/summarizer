from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand

from .config import Settings
from .handlers import build_router
from .llm import LLMClient
from .storage import MessageStore
from .summarizer import Summarizer


async def run() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()  # pyright: ignore[reportCallIssue]
    # No default parse_mode on purpose: service replies are plain text (and may
    # legitimately contain "<" / ">"), while model output is delivered through
    # rich messages, which carry their own Markdown.
    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    store = MessageStore()
    llm_client = LLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        fallback_models=settings.parsed_openai_fallback_models,
    )
    summarizer = Summarizer(llm_client, max_chars=settings.summary_max_chars)
    dispatcher.include_router(build_router(store, summarizer, settings))

    await bot.set_my_commands(
        [
            BotCommand(command="start", description="Start the bot"),
            BotCommand(command="help", description="Show help"),
            BotCommand(command="summary", description="Summarize recent messages"),
            BotCommand(command="ask", description="Ask a question (with optional web search)"),
        ]
    )

    await dispatcher.start_polling(bot)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
