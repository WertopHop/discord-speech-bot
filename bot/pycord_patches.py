"""Runtime patches for pycord 2.8.x voice receive (DAVE is mandatory server-side).

Root cause of "bot hears noise" (found via diag/last_utterance.wav + davey stats):
pycord's PacketDecryptor.decrypt_rtp() DAVE-decrypts the payload correctly, but
then calls packet.update_extended_header() on the DECRYPTED OPUS AUDIO. That
function parses the first bytes of the opus frame as an RTP extension header
and returns an offset (~8-12), so pycord slices 8-12 bytes off EVERY opus
frame. Opus then decodes the mangled frames into loud noise (or fails with
"corrupted stream" in ~26% of frames). DAVE decryption itself worked fine.

Fix: one clean receive chain in decrypt_rtp (transport AEAD -> DAVE -> final
opus payload) with no post-decrypt header re-parsing. PacketDecoder then only
opus-decodes.
"""
import logging
import os

import davey
import discord.opus
from discord.voice.packets.rtp import RTP_PACKET_TYPE_VOICE
from discord.voice.receive.reader import PacketDecryptor  # noqa: E402
from discord.voice.utils.dependencies import HAS_DAVEY

log = logging.getLogger(__name__)


_frame_stats = {"ok": 0, "dave_fail": 0, "opus_fail": 0, "enc": 0, "plain": 0, "nonvoice": 0}
_last_dave: tuple[object, int] | None = None  # (davey session, last sender user id)
_payload_dumps = {"raw": 0, "out": 0}


def _log_frame_stats() -> None:
    total = _frame_stats["ok"] + _frame_stats["dave_fail"] + _frame_stats["opus_fail"]
    if total and total % 200 == 0:
        log.info(
            "frame stats: ok=%d dave_fail=%d opus_fail=%d enc=%d plain=%d nonvoice=%d (total=%d)",
            _frame_stats["ok"],
            _frame_stats["dave_fail"],
            _frame_stats["opus_fail"],
            _frame_stats["enc"],
            _frame_stats["plain"],
            _frame_stats["nonvoice"],
            total,
        )
        if _last_dave is not None:
            dave, uid = _last_dave
            try:
                stats = dave.get_decryption_stats(uid, davey.MediaType.audio)
                # passthroughs==attempts -> frames passed through as plaintext
                log.info(
                    "davey decrypt[%s]: attempts=%s ok=%s fail=%s passthrough=%s",
                    uid,
                    stats.attempts,
                    stats.successes,
                    stats.failures,
                    stats.passthroughs,
                )
            except Exception:  # noqa: BLE001
                log.info("davey decrypt stats unavailable for user %s", uid)
            try:
                log.info(
                    "davey session: users=%s status=%s epoch=%s ready=%s",
                    dave.get_user_ids(),
                    dave.status,
                    dave.epoch,
                    dave.ready,
                )
            except Exception:  # noqa: BLE001
                pass


def _dump_payload(kind: str, payload: bytes) -> None:
    """Log the first raw/decrypted DAVE payloads for offline analysis."""
    if _payload_dumps[kind] >= 3:
        return
    _payload_dumps[kind] += 1
    log.info(
        "payload sample [%s] #%d len=%s head=%s tail=%s",
        kind,
        _payload_dumps[kind],
        len(payload),
        payload[:24].hex(),
        payload[-16:].hex(),
    )


def _dave_decrypt(dave, user_id: int, payload: bytes) -> bytes:
    """DAVE-decrypt a frame, tolerating passthrough senders.

    Solo senders (or senders whose E2EE transition never completed) keep
    sending plaintext frames. Per the DAVE spec, passthrough mode only passes
    through frames that FAIL the protocol frame check, so keeping it enabled
    permanently never breaks decryption of real E2EE frames.
    """
    if payload[-2:] == b"\xfa\xfa":
        _frame_stats["enc"] += 1
    else:
        _frame_stats["plain"] += 1
    _dump_payload("raw", payload)
    try:
        result = dave.decrypt(user_id, davey.MediaType.audio, payload)
    except Exception as exc:  # noqa: BLE001
        if "UnencryptedWhenPassthroughDisabled" not in str(exc):
            _frame_stats["dave_fail"] += 1
            _log_frame_stats()
            raise
        # Sender is in passthrough (solo user / pending transition). Enable it
        # for a very large number of transitions; encrypted frames, once they
        # start arriving, still decrypt normally.
        log.info("DAVE: sender is unencrypted (passthrough sender) - passthrough enabled")
        dave.set_passthrough_mode(True, 10_000)
        result = dave.decrypt(user_id, davey.MediaType.audio, payload)
    _dump_payload("out", result)
    _frame_stats["ok"] += 1
    _last_dave = (dave, user_id)
    return result


_decode_fail_logged = 0


def _log_decode_failure(payload: bytes, exc: Exception) -> None:
    """Log the first few undecodable frames with enough detail to debug DAVE.

    protocol_magic: DAVE protocol frames carry a 0xFAFA marker at the end -
    True means the frame is an encrypted protocol frame (keys issue),
    False means raw/passthrough audio (opus issue).
    INFO level: details belong in diag.log, not on the user's console.
    """
    global _decode_fail_logged
    _decode_fail_logged += 1
    if _decode_fail_logged > 3:
        return
    log.info(
        "frame decode failed #%d (%r) | len=%s head=%s tail=%s protocol_magic=%s",
        _decode_fail_logged,
        exc,
        len(payload),
        payload[:16].hex(),
        payload[-8:].hex(),
        b"\xfa\xfa" in payload[-6:],
    )


def _decode_pcm(decoder, payload: bytes, fec: bool) -> bytes:
    try:
        result = decoder.decode(payload, fec=fec)
    except Exception as exc:  # noqa: BLE001
        _frame_stats["opus_fail"] += 1
        _log_decode_failure(payload or b"", exc)
        _log_frame_stats()
        raise
    _frame_stats["ok"] += 1
    _log_frame_stats()
    return result


# ---------------------------------------------------------------------
# One clean receive chain: transport AEAD -> DAVE -> final opus payload.
# Pycord's decrypt_rtp DAVE-decrypts correctly but then re-parses an RTP
# extension header ON THE DECRYPTED OPUS AUDIO (update_extended_header on
# garbage returns an offset ~8-12) and slices those bytes off every frame
# -> speech becomes noise. Do the ext strip once, in the transport step.
# ---------------------------------------------------------------------
def _fixed_decrypt_rtp(self, packet):
    state = self.client._connection
    dave = state.dave_session if HAS_DAVEY else None
    if packet.payload != RTP_PACKET_TYPE_VOICE:
        # control datagrams (RTPFB/NACK, TWCC, ...) - decrypting them only
        # produces CryptoError spam on every reply; the reader skips b"" data
        _frame_stats["nonvoice"] += 1
        packet.decrypted_data = b""
        return b""
    data = self._decryptor_rtp(packet)  # transport AEAD (+ ext strip, once)
    if dave is not None and dave.ready:
        uid = state.ssrc_user_map.get(packet.ssrc)
        if uid:
            data = _dave_decrypt(dave, uid, data)
    packet.decrypted_data = data
    return data


PacketDecryptor.decrypt_rtp = _fixed_decrypt_rtp


def _fixed_decode_packet(self, packet):
    """PacketDecoder._decode_packet: opus-decode the already-final payload.

    DAVE decryption happens exactly once, in decrypt_rtp (see above) - the
    payload arriving here is plain opus (or opus silence for lost frames).
    """
    assert self._decoder is not None
    assert self.sink.client

    if packet:
        pcm = _decode_pcm(self._decoder, packet.decrypted_data, fec=False)
    else:
        # lost packet: reconstruct the previous frame from the next one (FEC)
        next_packet = self._buffer.peek_next()
        if next_packet is not None:
            pcm = _decode_pcm(self._decoder, next_packet.decrypted_data, fec=True)
        else:
            pcm = self._decoder.decode(None, fec=False)

    return packet, pcm


discord.opus.PacketDecoder._decode_packet = _fixed_decode_packet

# Normal MLS/DAVE participation is kept: an earlier experiment skipped the
# MLS handshake entirely (forcing a downgrade), but the server then simply
# stops forwarding other users' audio to the bot, so this was reverted.

# ---------------------------------------------------------------------
# Replace pycord's JitterBuffer with a plain FIFO.
# JitterBuffer drops every buffered packet except the first one on each
# sequence gap (the "packets were lost being flushed" warning) and stalls
# delivery until the head packet is sequential. Speech capture loses
# ~half the frames that way. STT does not need jitter reordering - it
# needs every packet delivered exactly once, in arrival order.
# ---------------------------------------------------------------------
from discord.voice.utils.buffer import JitterBuffer  # noqa: E402


class FifoPacketBuffer(JitterBuffer):
    def __init__(self) -> None:
        super().__init__(max_size=150, pref_size=0, prefill=0)

    def push(self, packet) -> bool:
        self._buffer.append(packet)  # FIFO: arrival order
        if len(self._buffer) > self.max_size:
            self._buffer.pop(0)  # overflow: drop the oldest, keep fresh audio
        self._update_has_item()
        return True

    def _update_has_item(self) -> None:
        if self._buffer:
            self._has_item.set()
        else:
            self._has_item.clear()

    def pop(self, *, timeout: float | None = 0):
        if not self._has_item.wait(timeout):
            return None
        packet = self._buffer.pop(0)
        self._update_has_item()
        if packet is not None:
            self._last_tx_seq = packet.sequence
        return packet

    def peek(self, *, all: bool = False):
        return self._buffer[0] if self._buffer else None

    def peek_next(self):
        return self._buffer[0] if self._buffer else None

    def gap(self) -> int:
        return 0


_orig_decoder_init = discord.opus.PacketDecoder.__init__


def _fifo_decoder_init(self, router, ssrc) -> None:
    _orig_decoder_init(self, router, ssrc)
    self._buffer = FifoPacketBuffer()


discord.opus.PacketDecoder.__init__ = _fifo_decoder_init

# one bad packet must not kill the router loop
_orig_pop_data = discord.opus.PacketDecoder.pop_data


def _safe_pop_data(self, *, timeout: float = 0):
    try:
        return _orig_pop_data(self, timeout=timeout)
    except Exception:
        # details are logged by _log_decode_failure; skip and keep the router
        # loop alive (returning None is handled by the router)
        return None

# adjust_rtpsize() mutates the packet (header/data shifts). If it ever runs
# twice on the same packet, transport AEAD breaks -> make it idempotent.
from discord.voice.packets.rtp import RTPPacket  # noqa: E402

_orig_adjust_rtpsize = RTPPacket.adjust_rtpsize


def _idempotent_adjust_rtpsize(self) -> None:
    if self._rtpsize:
        return
    _orig_adjust_rtpsize(self)


RTPPacket.adjust_rtpsize = _idempotent_adjust_rtpsize

# Telemetry: dump the structure of packets that fail transport decryption
from discord.voice.receive.reader import PacketDecryptor  # noqa: E402

_orig_aead_decrypt = PacketDecryptor._decrypt_rtp_aead_xchacha20_poly1305_rtpsize
_aead_fail_logged = 0


def _fixed_aead_decrypt(self, packet):
    """Corrected AEAD receive: pycord always strips 8 bytes of plaintext,
    which is right only for packets WITH the rtpsize extension (its body).
    For packets without it that cut corrupts the opus frame."""
    global _aead_fail_logged
    try:
        packet.adjust_rtpsize()
        nonce = packet.nonce + b"\x00" * 20
        result = self.box.decrypt(packet.data, bytes(packet.header), nonce)
    except Exception as exc:
        if _aead_fail_logged < 3:
            _aead_fail_logged += 1
            log.warning(
                "transport AEAD failed (%r) | extended=%s cc=%s header=%s "
                "data_len=%s data_head=%s",
                exc,
                packet.extended,
                packet.cc,
                packet.header.hex(),
                len(packet.data),
                packet.data[:24].hex(),
            )
        raise
    if packet.extended:
        # strip the RTP extension body (it precedes the DAVE frame in the
        # plaintext); the offset comes from the real header, never hardcoded
        offset = packet.update_extended_header(result)
        return result[offset:] if offset else result
    return result


PacketDecryptor._decrypt_rtp_aead_xchacha20_poly1305_rtpsize = _fixed_aead_decrypt

# Telemetry: count incoming RTP packets so "Discord sends nothing" is visible
from discord.voice.receive.router import PacketRouter  # noqa: E402

_rtp_count = 0
_orig_feed_rtp = PacketRouter.feed_rtp


def _counting_feed_rtp(self, packet):
    global _rtp_count
    _rtp_count += 1
    if _rtp_count % 100 == 1:
        log.info("voice RTP: %d packets received so far", _rtp_count)
    return _orig_feed_rtp(self, packet)


PacketRouter.feed_rtp = _counting_feed_rtp

# Optional packet capture for wire-format analysis: run with VOICE_DEBUG=1
if os.environ.get("VOICE_DEBUG", "").lower() in {"1", "true", "yes"}:
    from . import voice_debug  # noqa: F401


_orig_stop_recording = None
try:
    from discord.sinks.errors import RecordingException
    from discord.voice.client import VoiceClient as _PycordVoiceClient

    # Teardown QoL: pycord's router thread calls stop_recording() on shutdown
    # even when recording already stopped -> RecordingException traceback
    _orig_stop_recording = _PycordVoiceClient.stop_recording

    _stop_trace_logged = False

    def _quiet_stop_recording(self):
        global _stop_trace_logged
        import traceback
        import warnings

        # one-shot caller trace: "voice recording stopped" mid-session
        # previously appeared without any visible reason
        if not _stop_trace_logged:
            _stop_trace_logged = True
            stack = "".join(traceback.format_stack(limit=6)[:-1])
            log.info("stop_recording called from:\n%s", stack)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                return _orig_stop_recording(self)
            except RecordingException:
                return None

    _PycordVoiceClient.stop_recording = _quiet_stop_recording
except Exception:  # noqa: BLE001 - patching is best-effort
    pass

discord.opus.PacketDecoder.pop_data = _safe_pop_data


