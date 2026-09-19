"""PCM utilities: Discord audio conversion, WAV handling, an ffmpeg decoder, and an AudioSource bridge.

Latency is critical, so:
- the Discord stream (48 kHz stereo s16le) is converted to 16 kHz mono using vectorization (NumPy);
- TTS audio is decoded to PCM via a streaming ffmpeg process (without writing to disk);
- playback occurs from a continuous buffer: the next sentence
is synthesized without pauses between sentences.
"""
from __future__ import annotations

import asyncio
import inspect
import io
import shutil
import wave

import discord
import numpy as np

# Discord: 48 kHz, stereo, s16le; frame = 20 ms
DISCORD_SAMPLE_RATE = 48000
DISCORD_CHANNELS = 2
BYTES_PER_SAMPLE = 2
FRAME_MS = 20
FRAME_SIZE = DISCORD_SAMPLE_RATE * DISCORD_CHANNELS * BYTES_PER_SAMPLE * FRAME_MS // 1000  # 3840 bytes

# VAD and STT work with 16 kHz
VAD_SAMPLE_RATE = 16000
DECIMATION = DISCORD_SAMPLE_RATE // VAD_SAMPLE_RATE

# Size of the PCM chunk fed to the source (80 ms)
DECODE_CHUNK = FRAME_SIZE * 4


def get_ffmpeg_path() -> str:
    """Path to ffmpeg: system or from the imageio-ffmpeg package."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ffmpeg not found. Install it system-wide or add "
            "imageio-ffmpeg to requirements.txt (it ships the binary automatically)."
        ) from exc


def stereo48k_to_mono16k(data: bytes) -> np.ndarray:
    """Discord PCM (48k stereo s16le) -> mono 16 kHz int16 (for VAD and STT)."""
    samples = np.frombuffer(data, dtype=np.int16)
    usable = len(samples) - (len(samples) % DISCORD_CHANNELS)
    mono = samples[:usable].reshape(-1, DISCORD_CHANNELS).astype(np.float32).mean(axis=1)
    usable = len(mono) - (len(mono) % DECIMATION)
    if usable > 0:
        mono = mono[:usable].reshape(-1, DECIMATION).mean(axis=1)
    return np.clip(mono, -32768.0, 32767.0).astype(np.int16)


def mono16k_to_wav(pcm: np.ndarray) -> bytes:
    """Mono 16 kHz int16 -> WAV bytes (for sending to STT)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.setframerate(VAD_SAMPLE_RATE)
        wav.writeframes(pcm.astype("<i2", copy=False).tobytes())
    return buf.getvalue()


class FFmpegPCMDecoder:
    """Asynchronous decoder: audio stream on stdin -> PCM 48 kHz stereo s16le on stdout.

    Accepts any format that ffmpeg understands (wav/mp3/ogg/...). For raw PCM
    specify input_format="s16le" (sample rate/channels are taken from tts_sample_rate/mono).
    """

    def __init__(self, ffmpeg_path: str, input_format: str | None = None, input_rate: int = 24000) -> None:
        args = [ffmpeg_path, "-hide_banner", "-loglevel", "error"]
        if input_format:
            args += ["-f", input_format, "-ar", str(input_rate), "-ac", "1"]
        args += [
            "-i", "pipe:0",
            "-f", "s16le",
            "-ar", str(DISCORD_SAMPLE_RATE),
            "-ac", str(DISCORD_CHANNELS),
            "-c:a", "pcm_s16le",
            "pipe:1",
        ]
        self._args = args
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def read_stderr(self) -> bytes:
        """Collects ffmpeg diagnostics (call after the output stream ended)."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return b""
        try:
            return await proc.stderr.read()
        except Exception:  # noqa: BLE001
            return b""

    async def feed(self, data: bytes) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(data)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass

    async def finish_input(self) -> None:
        proc = self._proc
        if proc is not None and proc.stdin is not None and not proc.stdin.is_closing():
            try:
                result = proc.stdin.write_eof()  # sync or async depending on Python
                if inspect.isawaitable(result):
                    await result
            except (BrokenPipeError, ConnectionResetError):
                pass

    async def read(self) -> bytes:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return b""
        return await proc.stdout.read(DECODE_CHUNK)

    async def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.returncode is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            proc.kill()
        try:
            await proc.wait()
        except Exception:  # noqa: BLE001
            pass


class PipelinedAudioSource(discord.AudioSource):
    """AudioSource that pulls decoded PCM from an asyncio queue.

    discord.py/pycord AudioPlayer runs in a separate thread and calls read()
    synchronously. This class bridges the player thread to async producers
    (LLM -> TTS -> ffmpeg chain) via run_coroutine_threadsafe.

    read() must return exactly FRAME_SIZE bytes (20 ms) or b"" to end playback.
    """

    READ_TIMEOUT = 30.0

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=64)
        self._buf = bytearray()
        self._eof = False
        self._stopped = False

    async def push(self, chunk: bytes | None) -> None:
        """Called from the event loop. None marks end of stream."""
        if self._stopped:
            return
        await self._queue.put(chunk)

    def _next_chunk(self) -> bytes | None:
        fut = asyncio.run_coroutine_threadsafe(self._queue.get(), self._loop)
        try:
            return fut.result(timeout=self.READ_TIMEOUT)
        except Exception:
            # Timeout or loop shutdown: stop playback instead of hanging
            return None

    def read(self) -> bytes:
        if self._stopped or self._eof:
            return b""
        while len(self._buf) < FRAME_SIZE:
            chunk = self._next_chunk()
            if chunk is None:
                self._eof = True
                break
            self._buf.extend(chunk)
        if len(self._buf) < FRAME_SIZE:
            return b""  # tail shorter than one frame is dropped (<20 ms)
        out = bytes(self._buf[:FRAME_SIZE])
        del self._buf[:FRAME_SIZE]
        return out

    def cleanup(self) -> None:
        # Called by the player thread after playback stops
        self._stopped = True
