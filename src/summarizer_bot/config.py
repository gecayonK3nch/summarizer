from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_FALLBACK_MODELS = [
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]
DEFAULT_FALLBACK_MODELS_CSV = ",".join(DEFAULT_FALLBACK_MODELS)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str = Field(alias="TELEGRAM_BOT_TOKEN")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="openai/gpt-oss-120b:free", alias="OPENAI_MODEL")
    openai_fallback_models: str = Field(
        default=DEFAULT_FALLBACK_MODELS_CSV,
        alias="OPENAI_FALLBACK_MODELS",
    )
    summary_max_messages: int = Field(default=20, alias="SUMMARY_MAX_MESSAGES")
    summary_max_chars: int = Field(default=20_000, alias="SUMMARY_MAX_CHARS")
    summary_temperature: float = Field(default=0.2, alias="SUMMARY_TEMPERATURE")

    @property
    def parsed_openai_fallback_models(self) -> list[str]:
        models = [model.strip() for model in self.openai_fallback_models.split(",") if model.strip()]
        return models or DEFAULT_FALLBACK_MODELS.copy()
