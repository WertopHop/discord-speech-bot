"""Speech pipeline: utterance -> STT -> LLM (stream) -> sentences -> TTS -> PCM -> playback.

Latency techniques:
- LLM tokens are streamed and split into sentences; each sentence goes to TTS
  immediately (2-3 TTS requests in flight) instead of waiting for full reply;
- TTS audio is decoded by ffmpeg into a continuous PCM stream, so sentences
  play back-to-back without gaps;
- barge-in: user speech stops playback and cancels pending TTS tasks.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import discord

from .audio.pcm import DECODE_CHUNK, FFmpegPCMDecoder, PipelinedAudioSource
from .config import Settings
from .conversation import ConversationStore
from .openrouter import OpenRouterClient, OpenRouterError

log = logging.getLogger(__name__)

# Chars that end a sentence (incl. CJK and newline)
_SENTENCE_ENDERS = ".!?\n;…。！？；"
_MIN_SENTENCE_CHARS = 10

# Playback push timeout: player thread must consume, otherwise it is gone
_PUSH_TIMEOUT = 10.0


def clean_for_tts(text: str) -> str:
    """Strip markdown noise that TTS would read aloud."""
    out = text
    for ch in ("*", "`", "_", "~", "#", ">"):
        out = out.replace(ch, "")
    return " ".join(out.split())


class SentenceSplitter:
    """Incremental splitter: LLM token stream -> ready sentences."""

    def __init__(self, min_chars: int = _MIN_SENTENCE_CHARS) -> None:
        self._buf: list[str] = []
        self._len = 0
        self._min = min_chars

    def feed(self, token: str) -> list[str]:
        out: list[str] = []
        for ch in token:
            self._buf.append(ch)
            self._len += 1
            if ch in _SENTENCE_ENDERS and self._len >= self._min:
                s = self._flush()
                if s:
                    out.append(s)
        return out

    def flush(self) -> str:
        if self._len == 0:
            return ""
        return self._flush()

    def _flush(self) -> str:
        s = clean_for_tts("".join(self._buf).strip())
        self._buf.clear()
        self._len = 0
        return s
