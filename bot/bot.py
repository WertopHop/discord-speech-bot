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
from .audio.pcm import PipelinedAudioSource, get_ffmpeg_path
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
            command_prefix="!",
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
        if self.pipeline is None:
            return
        if vc.is_recording():
            return  # already listening
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

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    @commands.command(name="join", aliases=["j"])
    async def join(self, ctx: commands.Context) -> None:
        """Join the author's voice channel and start listening."""
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.reply("Сначала зайди в голосовой канал.")
            return
        channel = ctx.author.voice.channel
        vc = ctx.guild.voice_client
        if vc is not None and vc.is_connected():
            if vc.channel != channel:
                await vc.move_to(channel)
        else:
            if vc is not None:
                await vc.disconnect(force=True)
            vc = await channel.connect(reconnect=True, timeout=30.0)
        self._start_capture(vc)
        await ctx.reply(f"Подключился к {channel.mention}. Говори!")

    @commands.command(name="leave", aliases=["l"])
    async def leave(self, ctx: commands.Context) -> None:
        """Leave the voice channel."""
        vc = ctx.guild.voice_client
        if vc is None or not vc.is_connected():
            await ctx.reply("Я и так не в голосовом канале.")
            return
        self._stop_capture(vc)
        await vc.disconnect(force=True)
        await ctx.reply("Отключился.")

    @commands.command(name="stop")
    async def stop(self, ctx: commands.Context) -> None:
        """Stop current playback immediately."""
        vc = ctx.guild.voice_client
        if self.pipeline is not None and vc is not None:
            self.pipeline.interrupt(vc)
        await ctx.reply("Остановил." if vc is not None else "Нет активного соединения.")

    @commands.command(name="reset")
    async def reset(self, ctx: commands.Context) -> None:
        """Clear conversation history for this channel."""
        if self.conversations is not None:
            self.conversations.reset((ctx.guild.id, ctx.channel.id))
        await ctx.reply("История диалога очищена.")

    @commands.command(name="say")
    async def say(self, ctx: commands.Context, *, text: str) -> None:
        """Debug command: synthesize and play arbitrary text."""
        vc = ctx.guild.voice_client
        if vc is None or not vc.is_connected():
            await ctx.reply("Сначала `!join`.")
            return
        if self.or_client is None:
            await ctx.reply("Клиент OpenRouter не готов.")
            return

        async def text_stream():
            for ch in text:
                yield ch

        loop = asyncio.get_running_loop()
        source = PipelinedAudioSource(loop)
        session = PlaybackSession(
            self.settings, self.or_client, self.ffmpeg_path, source, text_stream()
        )
        vc.play(source)
        await session.run()