"""Discord bot: voice chat assistant (STT -> LLM -> TTS via OpenRouter).

Library: py-cord (discord module). Voice receive is used via a custom Sink;
pycord prints a RuntimeWarning about DAVE E2EE on every start_recording call -
it is suppressed because guild voice channels still use the legacy path.
"""
from __future__ import annotations

import asyncio
import logging
import warnings

import discord
from discord.ext import commands

from .audio.capture import VoiceCaptureSink
from .audio.pcm import get_ffmpeg_path
from .config import Settings
from .conversation import ConversationStore
from .openrouter import OpenRouterClient
from .pipeline import PlaybackSession, SpeechPipeline

log = logging.getLogger(__name__)


class SpeechBot(commands.Bot):
    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        super().__init__(
            command_prefix=commands.when_mentioned_or(settings.command_prefix),
            intents=intents,
            help_command=None,
        )
        self.settings = settings
        self.or_client: OpenRouterClient | None = None
        self.conversations: ConversationStore | None = None
        self.pipeline: SpeechPipeline | None = None
        self.ffmpeg_path = ""
        self._sinks: dict[int, VoiceCaptureSink] = {}

    async def setup_hook(self) -> None:
        # Fail fast if ffmpeg is missing
        self.ffmpeg_path = get_ffmpeg_path()
        log.info("ffmpeg: %s", self.ffmpeg_path)
        self.or_client = OpenRouterClient(self.settings)
        await self.or_client.start()
        self.conversations = ConversationStore(self.settings)
        self.pipeline = SpeechPipeline(
            self.settings, self.or_client, self.conversations, self.ffmpeg_path
        )
        self.pipeline.start()

    async def close(self) -> None:
        if self.or_client is not None:
            await self.or_client.close()
        await super().close()

    async def on_ready(self) -> None:
        log.info("logged in as %s (id=%s)", self.user, self.user.id)

    # ------------------------------------------------------------------
    # Voice capture wiring
    # ------------------------------------------------------------------
    def _make_utterance_cb(self, vc: discord.VoiceClient):
        async def on_utterance(user_id: int, wav: bytes) -> None:
            member = vc.guild.get_member(user_id)
            name = member.display_name if member else f"user-{user_id}"
            if self.pipeline is not None:
                await self.pipeline.submit(vc, user_id, name, wav)

        return on_utterance

    def _start_capture(self, vc: discord.VoiceClient) -> None:
        if vc.guild.id in self._sinks or self.pipeline is None:
            return
        loop = asyncio.get_running_loop()
        s = self.settings
        interrupt_cb = (
            (lambda: loop.call_soon_threadsafe(self.pipeline.interrupt, vc))
            if s.barge_in
            else (lambda: None)
        )
        sink = VoiceCaptureSink(
            loop,
            silence_ms=s.silence_ms,
            min_speech_ms=s.min_speech_ms,
            preroll_ms=s.preroll_ms,
            max_utterance_ms=s.max_utterance_ms,
            barge_in_min_ms=s.barge_in_min_ms,
            vad_aggressiveness=s.vad_aggressiveness,
            on_utterance=self._make_utterance_cb(vc),
            is_bot_speaking=lambda: vc.is_playing(),
            on_interrupt=interrupt_cb,
        )
        self._sinks[vc.guild.id] = sink
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vc.start_recording(sink, self._recording_stopped)
        log.info("voice capture started in #%s", vc.channel)

    def _stop_capture(self, vc: discord.VoiceClient) -> None:
        self._sinks.pop(vc.guild.id, None)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if vc.is_recording():
                    vc.stop_recording()
        except Exception:
            pass

    @staticmethod
    def _recording_stopped(exception: Exception | None) -> None:
        if exception is not None:
            log.error("voice recording stopped with error: %r", exception)
        else:
            log.info("voice recording stopped")

    async def on_voice_state_update(
        self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState
    ) -> None:
        # Bot was disconnected from the voice channel externally
        if member.id == self.user.id and before.channel is not None and after.channel is None:
            sink = self._sinks.pop(before.channel.guild.id, None)
            if sink is not None:
                sink.clear()
                log.info("voice capture cleaned up (disconnected)")