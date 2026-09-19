"""Discord bot: voice chat assistant (STT -> LLM -> TTS via OpenRouter).

Library: py-cord (discord module). Voice receive is used via a custom Sink;
pycord prints a RuntimeWarning about DAVE E2EE on every start_recording call -
it is suppressed because guild voice channels still use the legacy path.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
import warnings

import discord
from discord.ext import commands
from discord.voice import VoiceClient as VoiceClient  # non-deprecated (pycord 2.7+)

from . import pycord_patches  # noqa: F401  (DAVE 0 + robust voice receive)

from .audio.capture import VoiceCaptureSink
from .audio.pcm import PipelinedAudioSource, get_ffmpeg_path
from .config import Settings
from .conversation import ConversationStore
from .markers import mark
from .openrouter import OpenRouterClient
from .pipeline import PlaybackSession, SpeechPipeline

log = logging.getLogger(__name__)

COMMAND_PREFIX = "sp!"


class SpeechBot(commands.Bot):
    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.members = True  # required to map voice packets to users
        super().__init__(
            command_prefix=COMMAND_PREFIX,
            intents=intents,
            help_command=None,
        )
        self.settings = settings
        self._last_stop_log = 0.0
        self.or_client: OpenRouterClient | None = None
        self.conversations: ConversationStore | None = None
        self.pipeline: SpeechPipeline | None = None
        self.ffmpeg_path = ""
        self._sinks: dict[int, VoiceCaptureSink] = {}
        self._register_text_commands()

    def _register_text_commands(self) -> None:
        """Pycord does not auto-register commands defined in a Bot subclass."""
        for name in ("join", "leave", "stop", "reset", "say"):
            raw = getattr(type(self), name, None)
            if raw is None or self.get_command(name) is not None:
                continue
            self.add_command(
                commands.Command(
                    functools.partial(raw.callback, self),
                    name=raw.name,
                    aliases=raw.aliases,
                    help=raw.help,
                )
            )

    def _pipeline_ready(self) -> bool:
        return self.or_client is not None and self.pipeline is not None

    async def close(self) -> None:
        if self.or_client is not None:
            await self.or_client.close()
        await super().close()

    async def on_ready(self) -> None:
        # pycord has no setup_hook, so the pipeline is initialized here (once)
        if self.or_client is None:
            try:
                self.ffmpeg_path = get_ffmpeg_path()
                log.info("ffmpeg: %s", self.ffmpeg_path)
                self.or_client = OpenRouterClient(self.settings)
                await self.or_client.start()
                self.conversations = ConversationStore(self.settings)
                self.pipeline = SpeechPipeline(
                    self.settings, self.or_client, self.conversations, self.ffmpeg_path
                )
                self.pipeline.start()
                log.info("pipeline ready")
                mark("bot started, waiting for sp!join")
            except Exception:
                log.exception("pipeline startup failed")
        log.info("logged in as %s (id=%s)", self.user, self.user.id)

    async def on_command_error(self, ctx: commands.Context, error: Exception) -> None:
        log.error("command error: %r", error)

    # ------------------------------------------------------------------
    # Voice capture wiring
    # ------------------------------------------------------------------
    def _make_utterance_cb(self, vc: VoiceClient):
        async def on_utterance(user_id: int, pcm) -> None:
            member = vc.guild.get_member(user_id)
            name = member.display_name if member else f"user-{user_id}"
            if self.pipeline is not None:
                await self.pipeline.submit(vc, user_id, name, pcm)

        return on_utterance

    def _make_stopped_cb(self, vc: VoiceClient):
        def on_stopped(exception: Exception | None) -> None:
            if exception is not None:
                log.error("voice recording stopped with error: %r", exception)
                return
            now = time.monotonic()
            if now - self._last_stop_log < 5.0:
                return  # pycord fires the stop callback more than once
            self._last_stop_log = now
            log.info("voice recording stopped")
            # pycord's receive reader can die mid-session (no error reported);
            # the bot must not go deaf while still connected to the channel
            if vc.is_connected() and self._sinks.get(vc.guild.id):
                try:
                    log.info("voice capture: auto-restarting listener")
                    self._start_capture(vc)
                except Exception:
                    log.exception("voice capture auto-restart failed")

        return on_stopped

    def _start_capture(self, vc: VoiceClient) -> None:
        if self.pipeline is None:
            return
        if vc.is_recording():
            return  # already listening
        loop = asyncio.get_running_loop()
        s = self.settings
        # discard any previous sink cleanly (its watchdog thread must stop)
        old_sink = self._sinks.get(vc.guild.id)
        if old_sink is not None:
            old_sink.stop_watchdog()
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
        # pycord 2.8.x never calls sink.init() itself -> sink.vc stays None
        # and the receive router crashes on the first packet (assert self.sink.client)
        sink.init(vc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vc.start_recording(sink, self._make_stopped_cb(vc))
        log.info("voice capture started in #%s", vc.channel)

    def _stop_capture(self, vc: VoiceClient) -> None:
        self._sinks.pop(vc.guild.id, None)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if vc.is_recording():
                    vc.stop_recording()
        except Exception:
            pass

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
        if not self._pipeline_ready():
            await ctx.reply("Бот ещё инициализируется — подожди пару секунд и повтори.")
            return
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
            try:
                vc = await channel.connect(reconnect=True, timeout=30.0)
            except Exception as exc:
                log.exception("voice connect failed")
                await ctx.reply(f"Не смог подключиться к голосовому каналу: {exc!r}")
                return
        try:
            self._start_capture(vc)
        except Exception as exc:
            log.exception("failed to start voice capture")
            await ctx.reply(f"Не удалось запустить прослушку: {exc!r}")
            return
        conn = getattr(vc, "_connection", None)
        log.info(
            "DAVE negotiated version: %s | dave session: %s",
            getattr(conn, "dave_protocol_version", "?"),
            "active" if getattr(conn, "dave_session", None) else "none",
        )
        mark(f"joined {channel.name} - say a phrase")
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
            await ctx.reply("Сначала `sp!join`.")
            return
        if not self._pipeline_ready():
            await ctx.reply("Бот ещё инициализируется — подожди пару секунд и повтори.")
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