"""Streaming voice capture: per-user VAD segmentation and barge-in detection.

Subclasses pycord's Sink. write() is called from the voice receive thread with
VoiceData (decoded PCM, 48 kHz stereo s16le, ~20 ms per packet). Segmentation
runs inline; finalized utterances (WAV bytes) are handed to the event loop.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Awaitable, Callable

import numpy as np
from discord.sinks import Sink

from .pcm import mono16k_to_wav, stereo48k_to_mono16k
from .vad import VoiceActivityDetector

log = logging.getLogger(__name__)

UtteranceCallback = Callable[[int, bytes], Awaitable[None]]


class _SpeakerState:
    __slots__ = (
        "preroll",
        "speech",
        "speaking",
        "voiced_run",
        "silent_run",
        "interrupt_run",
    )

    def __init__(self, preroll_frames: int) -> None:
        self.preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
        self.speech: list[np.ndarray] = []
        self.speaking = False
        self.voiced_run = 0
        self.silent_run = 0
        self.interrupt_run = 0


class VoiceCaptureSink(Sink):
    """Per-user speech segmentation on top of pycord's voice receive."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        silence_ms: int,
        min_speech_ms: int,
        preroll_ms: int,
        max_utterance_ms: int,
        barge_in_min_ms: int,
        vad_aggressiveness: int,
        on_utterance: UtteranceCallback,
        is_bot_speaking: Callable[[], bool],
        on_interrupt: Callable[[], None],
    ) -> None:
        super().__init__()
        self._loop = loop
        self._on_utterance = on_utterance
        self._is_bot_speaking = is_bot_speaking
        self._on_interrupt = on_interrupt
        self._vad = VoiceActivityDetector(vad_aggressiveness)
        self._preroll_frames = max(1, preroll_ms // 20)
        self._min_speech_frames = max(1, min_speech_ms // 20)
        self._silence_frames = max(1, silence_ms // 20)
        self._max_frames = max(1, max_utterance_ms // 20)
        self._interrupt_frames = max(1, barge_in_min_ms // 20)
        self._states: dict[int, _SpeakerState] = {}
        self._interrupt_fired = False

    # PacketDecoder queries this: False -> opus is decoded to PCM for us
    def is_opus(self) -> bool:
        return False

    def write(self, data, user) -> None:
        # data is VoiceData (pycord) or raw bytes (fallback)
        pcm = data.pcm if hasattr(data, "pcm") else bytes(data)
        if not pcm or user is None or not hasattr(user, "id"):
            return
        uid = int(user.id)
        state = self._states.get(uid)
        if state is None:
            state = self._states[uid] = _SpeakerState(self._preroll_frames)
        try:
            mono = stereo48k_to_mono16k(pcm)
        except Exception:
            log.exception("audio frame conversion failed")
            return
        try:
            bot_speaking = bool(self._is_bot_speaking())
        except Exception:
            bot_speaking = False

        step = self._vad.frame_samples
        for i, voiced in enumerate(self._vad.process(mono)):
            chunk = mono[i * step : (i + 1) * step]
            self._advance(uid, state, chunk, voiced, bot_speaking)

    def _advance(
        self,
        uid: int,
        state: _SpeakerState,
        chunk: np.ndarray,
        voiced: bool,
        bot_speaking: bool,
    ) -> None:
        # Barge-in: sustained speech while the bot is talking
        if bot_speaking and voiced:
            state.interrupt_run += 1
            if state.interrupt_run >= self._interrupt_frames and not self._interrupt_fired:
                self._interrupt_fired = True
                try:
                    self._on_interrupt()
                except Exception:
                    log.exception("interrupt callback failed")
        else:
            state.interrupt_run = 0
            if not bot_speaking:
                self._interrupt_fired = False

        state.preroll.append(chunk)

        if state.speaking:
            state.speech.append(chunk)
            if voiced:
                state.silent_run = 0
            else:
                state.silent_run += 1
            if state.silent_run >= self._silence_frames or len(state.speech) >= self._max_frames:
                self._finalize(uid, state)
        elif voiced:
            state.voiced_run += 1
            if state.voiced_run >= self._min_speech_frames:
                # Utterance starts: pre-roll + current speech become one buffer
                state.speech = list(state.preroll)
                state.speaking = True
                state.silent_run = 0
        else:
            state.voiced_run = 0

    def _finalize(self, uid: int, state: _SpeakerState) -> None:
        state.speaking = False
        state.voiced_run = 0
        state.silent_run = 0
        frames = state.speech
        state.speech = []
        if len(frames) < self._min_speech_frames:
            return
        pcm = np.concatenate(frames)
        wav = mono16k_to_wav(pcm)
        asyncio.run_coroutine_threadsafe(self._on_utterance(uid, wav), self._loop)

    def drop_user(self, user_id: int) -> None:
        self._states.pop(user_id, None)

    def clear(self) -> None:
        self._states.clear()
