"""
Dialogue history and message assembly for LLMs.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Sequence

from .config import Settings

DEFAULT_SYSTEM_PROMPT = (
    "Your name is Moishe. You are a voice assistant in a Discord voice channel."
    "Personality: brazen, cheeky, and a bit of a lout—respond like a street-smart tough guy:"
    "use the informal 'you,' throw in some jabs, and skip the formalities or polite phrasing,"
    "but stay on topic; rudeness and insults are allowed—it's all part of the brash humor,"
    "Always reply in the same language the other person just used."
    "Keep answers short and conversational: 1–3 sentences,"
    "no markdown, lists, or emojis—the text is meant for text-to-speech (TTS) output."
    "Incoming messages follow the format 'Name: text'."
)


def build_system_prompt() -> str:
    """TODO: place for system prompt calibration."""
    return DEFAULT_SYSTEM_PROMPT


def preprocess_user_text(text: str) -> str:
    """TODO: place for normalizing/calibrating STT text before sending to LLM."""
    return text


def postprocess_reply(text: str) -> str:
    """TODO: place for processing LLM response before TTS (trimming, replacements, etc.)."""
    return text


class ConversationStore:
    """Dialogue history and message assembly for LLMs."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._history: dict[Hashable, list[dict[str, str]]] = defaultdict(list)

    def build_messages(
        self, key: Hashable, display_name: str, user_text: str
    ) -> list[dict[str, str]]:
        content = f"{display_name}: {preprocess_user_text(user_text)}"
        return [
            {"role": "system", "content": build_system_prompt()},
            *self._history[key],
            {"role": "user", "content": content},
        ]

    def add_exchange(
        self, key: Hashable, display_name: str, user_text: str, assistant_text: str
    ) -> None:
        if not assistant_text.strip():
            return
        history = self._history[key]
        history.append(
            {"role": "user", "content": f"{display_name}: {preprocess_user_text(user_text)}"}
        )
        history.append({"role": "assistant", "content": postprocess_reply(assistant_text)})
        limit = max(1, self._s.history_limit) * 2
        if len(history) > limit:
            del history[:-limit]

    def reset(self, key: Hashable) -> None:
        self._history.pop(key, None)

    def snapshot(self, key: Hashable) -> Sequence[dict[str, str]]:
        return tuple(self._history.get(key, ()))
