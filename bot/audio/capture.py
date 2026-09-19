"""Streaming voice capture: per-user VAD segmentation and barge-in detection.

Subclasses pycord's Sink. write() is called from the voice receive thread with
VoiceData (decoded PCM, 48 kHz stereo s16le, ~20 ms per packet). Segmentation
runs inline; finalized utterances (WAV bytes) are handed to the event loop.

Discord stops sending RTP while a user is silent (DTX), so utterances are
finalized by a watchdog thread after `silence_ms` without new audio - not by
counting silent frames.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Awaitable, Callable

import numpy as np
from discord.sinks import Sink

from .pcm import stereo48k_to_mono16k
from .vad import VoiceActivityDetector

log = logging.getLogger(__name__)

# The utterance payload is raw 16 kHz mono PCM (int16 numpy); the pipeline
# merges consecutive fragments and encodes one WAV per STT request.
UtteranceCallback = Callable[[int, "np.ndarray"], Awaitable[None]]


class _SpeakerState:
    __slots__ = (
        "preroll",
        "speech",
        "speaking",
        "voiced_run",
        "silent_run",
        "interrupt_run",
        "last_seen",
        "frames_total",
        "voiced_total",
    )

    def __init__(self, preroll_frames: int) -> None:
        self.preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
        self.speech: list[np.ndarray] = []
        self.speaking = False
        self.voiced_run = 0
        self.silent_run = 0
        self.interrupt_run = 0
        self.last_seen = 0.0
        self.frames_total = 0
        self.voiced_total = 0


class VoiceCaptureSink(Sink):
    """Per-user speech segmentation on top of pycord's voice receive."""

    # pycord 2.8.x router expects these on custom sinks but never defines them
    __sink_listeners__: tuple = ()

    def walk_children(self):
        return []

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
        self._logged_users: set[int] = set()
        self._warned_no_user = False
        self._lock = threading.Lock()
        self._stop_watchdog = threading.Event()
        self._silence_sec = silence_ms / 1000.0
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="voice-capture-watchdog", daemon=True
        )
        self._watchdog.start()

    # PacketDecoder queries this: False -> opus is decoded to PCM for us
    def is_opus(self) -> bool:
        return False

    def write(self, data, user) -> None:
        # data is VoiceData (pycord) or raw bytes (fallback)
        pcm = data.pcm if hasattr(data, "pcm") else bytes(data)
        if not pcm or user is None or not hasattr(user, "id"):
            if not self._warned_no_user:
                self._warned_no_user = True
                log.warning(
                    "voice packet without user mapping - check that "
                    "Server Members Intent is enabled in the developer portal"
                )
            return
        uid = int(user.id)
        with self._lock:
            state = self._states.get(uid)
            if state is None:
                state = self._states[uid] = _SpeakerState(self._preroll_frames)
            state.last_seen = time.monotonic()
        if uid not in self._logged_users:
            self._logged_users.add(uid)
            log.info("receiving voice from user id=%s", uid)
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
        decisions = self._vad.process(mono)
        with self._lock:
            state.frames_total += len(decisions)
            for i, voiced in enumerate(decisions):
                chunk = mono[i * step : (i + 1) * step]
                if voiced:
                    state.voiced_total += 1
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
        log.info("utterance captured: user=%s, %.1fs", uid, len(frames) * 0.02)
        asyncio.run_coroutine_threadsafe(self._on_utterance(uid, pcm), self._loop)

    def _watchdog_loop(self) -> None:
        """Finalizes utterances after silence_ms without packets (Discord DTX)."""
        last_stats = 0.0
        while not self._stop_watchdog.wait(0.1):
            now = time.monotonic()
            with self._lock:
                for uid, state in list(self._states.items()):
                    if state.speaking and (now - state.last_seen) >= self._silence_sec:
                        self._finalize(uid, state)
            # Periodic VAD stats so speech detection is visible in the console
            if now - last_stats >= 2.0:
                last_stats = now
                with self._lock:
                    for uid, state in self._states.items():
                        if state.frames_total > 0 and (now - state.last_seen) < 5.0:
                            log.info(
                                "VAD stats: user=%s voiced=%d/%d frames",
                                uid,
                                state.voiced_total,
                                state.frames_total,
                            )
                            state.frames_total = 0
                            state.voiced_total = 0

    def stop_watchdog(self) -> None:
        """Stop the watchdog thread (called when the sink is replaced)."""
        self._stop_watchdog.set()

    def drop_user(self, user_id: int) -> None:
        with self._lock:
            self._states.pop(user_id, None)

    def clear(self) -> None:
        with self._lock:
            self._states.clear()
        self._logged_users.clear()
        self._stop_watchdog.set()
