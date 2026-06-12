# Summarizer Bot

Telegram bot that collects recent chat messages and produces a short AI-generated summary for the last N messages.

The default model is a strong free OpenRouter-compatible option: `openai/gpt-oss-120b:free`.

## What it does

- Listens for regular text messages in chats where the bot is present.
- Stores recent messages in memory per chat.
- Summarizes the last N messages with an OpenAI-compatible model.
- Exposes a simple `/summary N` command.
- Can answer arbitrary questions via `/ask`, automatically deciding whether to provide chat context or to search the internet (via DuckDuckGo) for real-time information.

## Quick start

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -e .[dev]
```

3. Copy `.env.example` to `.env` and fill in your tokens.
4. Run the bot:

```bash
python -m summarizer_bot.main
```

## Commands

- `/start` - short intro
- `/help` - usage notes
- `/summary 10` - summarize the last 10 stored text messages
- `/ask <question>` or `!ask <question>` - Ask the bot any question. The bot will automatically analyze if it needs to search the internet (using DuckDuckGo) or refer to the recent chat history to answer your question.

## Configuration

Environment variables:

- `TELEGRAM_BOT_TOKEN` - Telegram bot token from BotFather
- `OPENAI_API_KEY` - API key for the LLM provider, such as OpenRouter
- `OPENAI_BASE_URL` - API base URL for OpenAI-compatible providers. Default: `https://openrouter.ai/api/v1`
- `OPENAI_MODEL` - primary model for summaries. Default: `openai/gpt-oss-120b:free`
- `OPENAI_FALLBACK_MODELS` - comma-separated fallback models used when the primary one is unavailable
- `SUMMARY_MAX_MESSAGES` - upper limit for how many messages can be summarized at once
- `SUMMARY_MAX_CHARS` - maximum total character budget for the prompt context
- `SUMMARY_TEMPERATURE` - generation temperature for the summary

## Notes

- Message history is stored in memory only. Restarting the process clears recent chat context.
- While a command is being processed, the bot shows Telegram's "typing" status until it sends the response.
- The LLM client is intentionally OpenAI-compatible so you can point it at OpenAI or another compatible provider.
- If the primary model returns 404 or provider errors, the bot automatically tries `OPENAI_FALLBACK_MODELS`.
- If you want different free models, override `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_FALLBACK_MODELS` in `.env`.
