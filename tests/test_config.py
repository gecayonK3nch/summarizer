from summarizer_bot.config import Settings


def test_settings_parse_fallback_models_from_comma_string() -> None:
    settings = Settings(
        TELEGRAM_BOT_TOKEN="test-token",
        OPENAI_API_KEY="test-key",
        OPENAI_FALLBACK_MODELS="model-a, model-b,model-c",
    )

    assert settings.parsed_openai_fallback_models == ["model-a", "model-b", "model-c"]
