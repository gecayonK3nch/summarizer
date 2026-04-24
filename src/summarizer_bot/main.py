from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand

from .config import Settings
from .handlers import build_router
from .llm import LLMClient
from .storage import MessageStore
from .summarizer import Summarizer


async def run() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()  # pyright: ignore[reportCallIssue]
    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
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
        ]
    )

    await dispatcher.start_polling(bot)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
