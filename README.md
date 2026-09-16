# Summarizer Bot

Telegram bot that collects recent chat messages and produces a short AI-generated summary for the last N messages.

The default model is a strong free OpenRouter-compatible option: `openai/gpt-oss-120b:free`.

## What it does

- Listens for regular text messages in chats where the bot is present.
- Stores recent messages in memory per chat.
- Summarizes the last N messages with an OpenAI-compatible model.
- Exposes a simple `/summary N` command.
- Can answer arbitrary questions via `/ask`, using fast rule-based routing to decide whether to include recent chat history or search the internet (via DuckDuckGo) for real-time information.

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
- `/summary N` - summarize the last N stored text messages (default 10, capped by `SUMMARY_MAX_MESSAGES`)
- `/ask <question>` or `!ask <question>` - Ask the bot any question. The bot uses local rules to decide whether it needs recent chat history or a DuckDuckGo search before answering.

## Configuration

Environment variables:

- `TELEGRAM_BOT_TOKEN` - Telegram bot token from BotFather
- `OPENAI_API_KEY` - API key for the LLM provider, such as OpenRouter
- `OPENAI_BASE_URL` - API base URL for OpenAI-compatible providers. Default: `https://openrouter.ai/api/v1`
- `OPENAI_MODEL` - primary model for summaries. Default: `openai/gpt-oss-120b:free`
- `OPENAI_FALLBACK_MODELS` - comma-separated fallback models used when the primary one is unavailable. The default chain is shorter now to avoid long retry cascades.
- `SUMMARY_MAX_MESSAGES` - upper limit for how many messages can be summarized at once
- `SUMMARY_MAX_CHARS` - maximum total character budget for the prompt context
- `SUMMARY_TEMPERATURE` - generation temperature for the summary

## Rich formatting

Answers and summaries are sent as Telegram **rich messages** (Bot API 10.1+), so the
model's Markdown is preserved and rendered natively — no lossy post-processing:

- Headings, **bold**/*italic*/~~strikethrough~~, bullet/numbered/task lists, and `> blockquotes`.
- GitHub-style tables (`| col | col |`).
- Fenced code blocks with language highlighting (```` ```python … ``` ````).
- LaTeX math: inline `$...$` and display `$$...$$`, rendered by the Telegram client itself.

The raw model output goes straight into `InputRichMessage(markdown=...)` via
`sendRichMessage`. A single rich message holds up to 32768 characters. If a rich send
ever fails (malformed markup or an oversized payload), the bot falls back to a plain-text reply.

Requires `aiogram>=3.30` (Bot API 10.2). The older matplotlib PNG / Unicode-math
pipeline was removed, since the Telegram client now renders math and code directly.

## Notes

- Message history is stored in memory only. Restarting the process clears recent chat context.
- In group chats, disable Bot API privacy mode via BotFather (`/setprivacy`), otherwise the bot only receives commands and cannot collect messages for `/summary` or handle `!ask`.
- Service replies (`/start`, `/help`, error messages) are sent as plain text without a parse mode, so they can safely contain characters like `<` and `>`.
- While a command is being processed, the bot shows Telegram's "typing" status until it sends the response.
- The LLM client is intentionally OpenAI-compatible so you can point it at OpenAI or another compatible provider.
- The `/ask` command uses local rules for history/search routing, so it does not spend an extra LLM call just to decide whether context is needed.
- If the primary model returns 404 or provider errors, the bot automatically tries `OPENAI_FALLBACK_MODELS`.
- If you want different free models, override `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_FALLBACK_MODELS` in `.env`.
