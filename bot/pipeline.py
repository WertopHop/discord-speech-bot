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


class PlaybackSession:
    """One response playback: text stream -> TTS tasks -> PCM -> audio source."""

    MAX_PARALLEL_TTS = 3

    def __init__(
        self,
        settings: Settings,
        or_client: OpenRouterClient,
        ffmpeg_path: str,
        source: PipelinedAudioSource,
        text_iter: AsyncIterator[str],
    ) -> None:
        self._s = settings
        self._or = or_client
        self._ff = ffmpeg_path
        self.source = source
        self._text_iter = text_iter
        self.stop_event = asyncio.Event()
        self._tts_sem = asyncio.Semaphore(self.MAX_PARALLEL_TTS)

    def stop(self) -> None:
        self.stop_event.set()

    async def run(self) -> bool:
        """Runs the whole chain. Returns True if it finished uncancelled."""
        try:
            await self._produce()
        except Exception:
            log.exception("playback pipeline error")
        finally:
            await self._safe_push(None)
        return not self.stop_event.is_set()

    async def _safe_push(self, chunk: bytes | None) -> None:
        try:
            await asyncio.wait_for(self.source.push(chunk), timeout=_PUSH_TIMEOUT)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # Player thread is not consuming anymore - stop producing
            self.stop_event.set()

    async def _emit(self, pcm: bytes) -> None:
        if not pcm or self.stop_event.is_set():
            return
        for off in range(0, len(pcm), DECODE_CHUNK):
            if self.stop_event.is_set():
                return
            await self._safe_push(pcm[off : off + DECODE_CHUNK])
            if self.stop_event.is_set():
                return

    async def _produce(self) -> None:
        sent_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=6)
        task_q: asyncio.Queue[asyncio.Task | None] = asyncio.Queue(maxsize=self.MAX_PARALLEL_TTS)

        async def sentence_producer() -> None:
            splitter = SentenceSplitter()
            try:
                async for token in self._text_iter:
                    if self.stop_event.is_set():
                        return
                    for s in splitter.feed(token):
                        await sent_q.put(s)
            finally:
                last = splitter.flush()
                if last:
                    try:
                        await sent_q.put(last)
                    except asyncio.CancelledError:
                        pass
                await sent_q.put(None)

        async def spawner() -> None:
            while True:
                s = await sent_q.get()
                if s is None or self.stop_event.is_set():
                    await task_q.put(None)
                    return
                await task_q.put(asyncio.create_task(self._tts_pcm(s)))

        producer = asyncio.create_task(sentence_producer())
        spawner_task = asyncio.create_task(spawner())
        try:
            while True:
                task = await task_q.get()
                if task is None or self.stop_event.is_set():
                    break
                # FIFO await keeps sentence order intact
                pcm = await task
                await self._emit(pcm)
        finally:
            producer.cancel()
            spawner_task.cancel()
            await asyncio.gather(producer, spawner_task, return_exceptions=True)

    async def _tts_pcm(self, text: str) -> bytes:
        """TTS one sentence -> full PCM (48 kHz stereo s16le)."""
        async with self._tts_sem:
            if self.stop_event.is_set():
                return b""
            input_format = "s16le" if self._s.tts_format in {"pcm", "pcm16"} else None
            decoder = FFmpegPCMDecoder(
                self._ff, input_format=input_format, input_rate=self._s.tts_sample_rate
            )
            pcm = bytearray()
            await decoder.start()

            async def pump() -> None:
                try:
                    async for chunk in self._or.stream_tts(text, self.stop_event):
                        await decoder.feed(chunk)
                finally:
                    await decoder.finish_input()

            pump_task = asyncio.create_task(pump())
            try:
                while True:
                    chunk = await decoder.read()
                    if not chunk:
                        break
                    pcm.extend(chunk)
                    if self.stop_event.is_set():
                        break
            finally:
                pump_task.cancel()
                await asyncio.gather(pump_task, return_exceptions=True)
                await decoder.close()
            return b"" if self.stop_event.is_set() else bytes(pcm)


class SpeechPipeline:
    """Serializes utterances per process: STT -> LLM -> TTS playback."""

    def __init__(
        self,
        settings: Settings,
        or_client: OpenRouterClient,
        conversations: ConversationStore,
        ffmpeg_path: str,
    ) -> None:
        self._s = settings
        self._or = or_client
        self._conv = conversations
        self._ff = ffmpeg_path
        self._queue: asyncio.Queue[tuple[discord.VoiceClient, int, str, bytes]] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._session: PlaybackSession | None = None
        # Increments on barge-in; stale processing aborts itself
        self._generation = 0

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run(), name="speech-pipeline")

    async def submit(
        self, vc: discord.VoiceClient, user_id: int, display_name: str, wav: bytes
    ) -> None:
        await self._queue.put((vc, user_id, display_name, wav))

    def interrupt(self, vc: discord.VoiceClient) -> None:
        """Barge-in: stop playback and cancel the in-flight reply."""
        self._generation += 1
        if self._session is not None:
            self._session.stop()
        if vc.is_playing():
            vc.stop()

    async def _run(self) -> None:
        while True:
            vc, user_id, display_name, wav = await self._queue.get()
            try:
                await self._process(vc, user_id, display_name, wav)
            except Exception:
                log.exception("utterance processing failed")
            finally:
                self._queue.task_done()

    async def _process(
        self, vc: discord.VoiceClient, user_id: int, display_name: str, wav: bytes
    ) -> None:
        if not vc.is_connected():
            return
        gen = self._generation

        text = await self._or.transcribe(wav)
        if not text:
            log.info("STT: empty transcript (user=%s)", user_id)
            return
        if gen != self._generation:
            return  # barge-in happened during STT
        log.info("STT [%s]: %s", display_name, text)

        key = (vc.guild.id, vc.channel.id if vc.channel else 0)
        messages = self._conv.build_messages(key, display_name, text)

        collected: list[str] = []
        completed = [False]

        async def llm_tokens() -> AsyncIterator[str]:
            parts: list[str] = []
            async for token in self._or.stream_chat(messages):
                parts.append(token)
                yield token
            completed[0] = True
            collected.append("".join(parts))

        loop = asyncio.get_running_loop()
        source = PipelinedAudioSource(loop)
        session = PlaybackSession(self._s, self._or, self._ff, source, llm_tokens())
        self._session = session
        try:
            vc.play(source)
            played = await session.run()
        finally:
            self._session = None

        if played and completed[0] and gen == self._generation:
            reply = collected[0] if collected else ""
            log.info("LLM -> [%s]: %s", display_name, reply)
            self._conv.add_exchange(key, display_name, text, reply)
        elif session.stop_event.is_set():
            log.info("reply cancelled (barge-in or !stop)")
