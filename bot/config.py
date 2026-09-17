"""Bot configuration: reading and validating environment variables from .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(RuntimeError):
    """Configuration error: missing required or invalid environment variables."""


def _str(name: str, default: str) -> str:
    raw = (os.getenv(name) or "").strip()
    return raw if raw else default


def _int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}: expected integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}: expected number, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "да"}


@dataclass(frozen=True, slots=True)
class Settings:
    """All bot settings in one place (filled from .env)."""

    discord_token: str
    openrouter_api_key: str
    openrouter_base_url: str

    stt_model: str
    llm_model: str
    tts_model: str
    tts_voice: str
    tts_format: str
    tts_sample_rate: int

    llm_system_prompt: str
    llm_max_tokens: int
    llm_temperature: float
    history_limit: int

    vad_aggressiveness: int
    silence_ms: int
    min_speech_ms: int
    preroll_ms: int
    max_utterance_ms: int

    barge_in: bool
    barge_in_min_ms: int

    command_prefix: str
    request_timeout: float


def get_settings() -> Settings:
    """Reads .env and validates. Raises ConfigError with a clear message."""
    missing = [
        name
        for name in ("DISCORD_TOKEN", "OPENROUTER_API_KEY")
        if not (os.getenv(name) or "").strip()
    ]
    if missing:
        raise ConfigError(
            "Required environment variables are not set: "
            + ", ".join(missing)
            + "\nPlease copy .env-example to .env and fill in the values."
        )

    settings = Settings(
        # Secrets
        discord_token=(os.getenv("DISCORD_TOKEN") or "").strip(),
        openrouter_api_key=(os.getenv("OPENROUTER_API_KEY") or "").strip(),
        openrouter_base_url=_str(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/"),
        # Models
        stt_model=_str("STT_MODEL", "openai/whisper-large-v3-turbo"),
        llm_model=_str("LLM_MODEL", "qwen/qwen3-30b-a3b-instruct-2507"),
        tts_model=_str("TTS_MODEL", "x-ai/grok-voice-tts-1.0"),
        tts_voice=_str("TTS_VOICE", "Eve"),
        tts_format=_str("TTS_FORMAT", "wav").lower(),
        tts_sample_rate=_int("TTS_SAMPLE_RATE", 24000),
        # LLM
        llm_system_prompt=os.getenv("LLM_SYSTEM_PROMPT", "") or "",
        llm_max_tokens=_int("LLM_MAX_TOKENS", 512),
        llm_temperature=_float("LLM_TEMPERATURE", 0.7),
        history_limit=_int("HISTORY_LIMIT", 12),
        # VAD
        vad_aggressiveness=_int("VAD_AGGRESSIVENESS", 2),
        silence_ms=_int("SILENCE_MS", 700),
        min_speech_ms=_int("MIN_SPEECH_MS", 200),
        preroll_ms=_int("PREROLL_MS", 300),
        max_utterance_ms=_int("MAX_UTTERANCE_MS", 30000),
        # Barge-in
        barge_in=_bool("BARGE_IN", True),
        barge_in_min_ms=_int("BARGE_IN_MIN_MS", 350),
        # Other
        command_prefix=_str("COMMAND_PREFIX", "!"),
        request_timeout=_float("REQUEST_TIMEOUT", 60.0),
    )

    if settings.tts_format == "pcm" and not 8000 <= settings.tts_sample_rate <= 48000:
        raise ConfigError("TTS_SAMPLE_RATE: allowed 8000..48000 Hz")

    return settings
