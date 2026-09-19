"""OpenRouter client.

Three endpoints:
- POST /audio/transcriptions - STT (base64 WAV -> text)
- POST /chat/completions - LLM (SSE stream of tokens)
- POST /audio/speech - TTS (text -> raw audio byte stream)

All requests go through one aiohttp session (connection reuse = lower latency).
"""
from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
from typing import Any, AsyncIterator, Sequence

import aiohttp

from .config import Settings

log = logging.getLogger(__name__)


class OpenRouterError(RuntimeError):
    """Error accessing OpenRouter (network, HTTP, response format)."""


_EXTRA_HEADERS = {
    # OpenRouter recommends specifying the application (optional)
    "HTTP-Referer": "https://localhost",
    "X-Title": "discord-speech-bot",
}

# TTS is per short sentence - a hung provider request must not freeze a reply
TTS_TOTAL_TIMEOUT = 30.0


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            return
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=15.0,
            sock_read=self._s.request_timeout,
        )
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._s.openrouter_api_key}",
                **_EXTRA_HEADERS,
            },
        )

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def _url(self, path: str) -> str:
        return f"{self._s.openrouter_base_url}{path}"

    # ------------------------------------------------------------------
    # Internal assistants
    # ------------------------------------------------------------------
    async def _raise_for_status(self, resp: aiohttp.ClientResponse, what: str) -> None:
        if resp.status != 200:
            body = await resp.text()
            raise OpenRouterError(f"{what}: HTTP {resp.status}: {body[:400]}")

    @staticmethod
    async def _release(resp: aiohttp.ClientResponse) -> None:
        """Carefully return the connection to the pool (release can be sync/async)."""
        if resp.closed:
            return
        try:
            result = resp.release()
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass

    # ------------------------------------------------------------------
    # STT: /audio/transcriptions
    # ------------------------------------------------------------------
    async def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str:
        """Speech -> text (WAV 16 kHz mono s16le). Empty string if no speech."""
        payload = {
            "model": self._s.stt_model,
            "input_audio": {
                "data": base64.b64encode(wav_bytes).decode("ascii"),
                "format": "wav",
            },
        }
        if language:
            payload["language"] = language
        assert self._session is not None
        try:
            async with self._session.post(
                self._url("/audio/transcriptions"),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._s.request_timeout),
            ) as resp:
                if resp.status == 400 and language:
                    # endpoint may not accept the language hint - retry without it
                    log.info("STT: language hint rejected, retrying without it")
                    payload.pop("language", None)
                    async with self._session.post(
                        self._url("/audio/transcriptions"),
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self._s.request_timeout),
                    ) as resp2:
                        await self._raise_for_status(resp2, "audio/transcriptions")
                        data: dict[str, Any] = await resp2.json(content_type=None)
                else:
                    await self._raise_for_status(resp, "audio/transcriptions")
                    data = await resp.json(content_type=None)
        except asyncio.TimeoutError as exc:
            raise OpenRouterError("STT: timeout request") from exc
        return str(data.get("text") or "").strip()

    # ------------------------------------------------------------------
    # LLM: /chat/completions (SSE-stream)
    # ------------------------------------------------------------------
    async def stream_chat(self, messages: Sequence[dict[str, str]]) -> AsyncIterator[str]:
        """Streams tokens of the LLM response (content only)."""
        payload = {
            "model": self._s.llm_model,
            "messages": list(messages),
            "stream": True,
            "max_tokens": self._s.llm_max_tokens,
            "temperature": self._s.llm_temperature,
        }
        assert self._session is not None
        resp = await self._session.post(self._url("/chat/completions"), json=payload)
        try:
            await self._raise_for_status(resp, "chat/completions")
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):
                    # strings starting with ":", - keep-alive comments OpenRouter
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content
        finally:
            await self._release(resp)

    # ------------------------------------------------------------------
    # TTS: /audio/speech (raw audio byte stream)
    # ------------------------------------------------------------------
    async def stream_tts(
        self, text: str, stop_event: asyncio.Event | None = None
    ) -> AsyncIterator[bytes]:
        """Text -> audio byte stream (format specified by settings.tts_format)."""
        payload = {
            "model": self._s.tts_model,
            "input": text,
            "voice": self._s.tts_voice,
            "response_format": self._s.tts_format,
        }
        assert self._session is not None
        # a hung TTS request must not freeze the whole reply
        resp = await self._session.post(
            self._url("/audio/speech"),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TTS_TOTAL_TIMEOUT),
        )
        try:
            await self._raise_for_status(resp, "audio/speech")
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type:
                body = await resp.text()
                raise OpenRouterError(f"TTS returned JSON instead of audio: {body[:300]}")
            received = 0
            async for chunk in resp.content.iter_chunked(8192):
                if stop_event is not None and stop_event.is_set():
                    break
                if chunk:
                    received += len(chunk)
                    yield chunk
            if received == 0:
                log.warning(
                    "TTS: empty response (status=%s, content-type=%s)",
                    resp.status,
                    resp.headers.get("Content-Type", "?"),
                )
            else:
                log.info(
                    "TTS response: %d bytes (content-type=%s)",
                    received,
                    resp.headers.get("Content-Type", "?"),
                )
        except asyncio.TimeoutError as exc:
            raise OpenRouterError("TTS: request timed out") from exc
        finally:
            if not resp.closed:
                resp.close()

