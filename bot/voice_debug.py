"""Packet capture + offline analysis for DAVE voice debugging.

Activated with VOICE_DEBUG=1 (env var). Captures the first ~400 raw voice
RTP datagrams and the transport key, then on process exit prints a summary:
how many packets decrypt with each candidate construction, whether the decoded
payloads are plain opus or DAVE frames, and packet structure samples.
Prints no key material.

Wire-format candidate constructions (module level for testability):
Aead.decrypt(ciphertext, aad, nonce).
"""
from __future__ import annotations

import atexit
import logging
import os

log = logging.getLogger(__name__)


def _parse_packet(raw: bytes) -> tuple[bool, bytes] | None:
    """Return (has_extension, payload_region) or None for tiny packets."""
    cc = raw[0] & 0xF
    ext = bool(raw[0] & 0x10)
    data = raw[12 + cc * 4:]
    if len(data) < 9:  # header + 4-byte nonce + at least 1 byte of payload
        return None
    return ext, data


def _v_rtpsize_ext(raw: bytes, data: bytes) -> tuple[bytes, bytes, bytes]:
    """RTPSIZE mode: 4-byte nonce suffix; AAD includes ext header if present."""
    ext = bool(raw[0] & 0x10)
    need = 9 if ext else 5
    if len(data) < need:
        raise ValueError("too short for rtpsize")
    return (
        data[4:-4] if ext else data[:-4],
        raw[:12] + (data[:4] if ext else b""),
        data[-4:] + b"\x00" * 20,
    )


def _v_rtpsize_plain(raw: bytes, data: bytes) -> tuple[bytes, bytes, bytes]:
    """RTPSIZE mode with header-only AAD (extension left in ciphertext)."""
    if len(data) < 5:
        raise ValueError("too short for rtpsize")
    return data[:-4], raw[:12], data[-4:] + b"\x00" * 20


def _v_suffix24(raw: bytes, data: bytes) -> tuple[bytes, bytes, bytes]:
    """Non-RTPSIZE AEAD mode: 24-byte nonce suffix."""
    if len(data) < 29:
        raise ValueError("too short for suffix24")
    return data[:-24], raw[:12], data[-24:]


_VARIANTS: dict[str, object] = {
    "rtpsize aad-with-ext (pycord model)": _v_rtpsize_ext,
    "rtpsize aad-header-only": _v_rtpsize_plain,
    "suffix-24 nonce (non-rtpsize mode)": _v_suffix24,
}

if os.environ.get("VOICE_DEBUG", "").lower() in {"1", "true", "yes"}:
    from discord.voice.receive.reader import PacketDecryptor

    # Show the MLS/DAVE gateway flow (key packages, commits, transitions)
    logging.getLogger("discord.voice.gateway").setLevel(logging.DEBUG)

    _LIMIT = 600
    _captured: list[tuple[int, int, bool, bytes]] = []  # ssrc, seq, extended, raw
    _key_holder: dict[str, bytes] = {}

    _orig_decrypt_rtp = PacketDecryptor.decrypt_rtp

    def _capturing_decrypt_rtp(self, packet):
        if len(_captured) < _LIMIT and not packet.decrypted_data:
            raw = bytes(packet.header) + bytes(packet.data)
            _captured.append((packet.ssrc, packet.sequence, packet.extended, raw))
        return _orig_decrypt_rtp(self, packet)

    PacketDecryptor.decrypt_rtp = _capturing_decrypt_rtp

    from discord.voice import gateway as voice_gateway

    _orig_load_secret_key = voice_gateway.VoiceWebSocket.load_secret_key

    async def _capturing_load_secret_key(self, data):
        _key_holder["key"] = bytes(data["secret_key"])
        return await _orig_load_secret_key(self, data)

    voice_gateway.VoiceWebSocket.load_secret_key = _capturing_load_secret_key

    def _finalize() -> None:
        try:
            _analyze()
        except Exception as exc:  # noqa: BLE001
            print(f"[voice-debug] analysis failed: {exc!r}")

    atexit.register(_finalize)

    def _analyze() -> None:
        from nacl.secret import Aead

        key = _key_holder.get("key", b"")
        if not key or not _captured:
            print("[voice-debug] nothing captured (no packets or no key)")
            return
        box = Aead(key)
        print(f"[voice-debug] captured {len(_captured)} packets, transport key captured")

        len_hist: dict[str, int] = {}
        b0_hist: dict[str, int] = {}
        cc_hist: dict[str, int] = {}
        ext_count = 0
        for _ssrc, _seq, _extended, raw in _captured:
            b0 = raw[0]
            ext_count += 1 if b0 & 0x10 else 0
            cc = str(b0 & 0xF)
            cc_hist[cc] = cc_hist.get(cc, 0) + 1
            tag = f"{b0:02x}"
            b0_hist[tag] = b0_hist.get(tag, 0) + 1
            n = len(raw)
            bucket = "<24" if n < 24 else "<64" if n < 64 else "<100" if n < 100 else ">=100"
            len_hist[bucket] = len_hist.get(bucket, 0) + 1
        print(f"[voice-debug] lengths: {len_hist}")
        print(
            f"[voice-debug] first-byte: {b0_hist} | cc: {cc_hist} | "
            f"ext-flag: {ext_count}/{len(_captured)}"
        )
        for _ssrc, seq, _extended, raw in _captured[:3]:
            tail = "..." if len(raw) > 56 else ""
            print(f"[voice-debug] raw: seq={seq} len={len(raw)} hex={raw[:56].hex()}{tail}")

        best_name = ""
        best_fn = None
        best_ok = 0
        for name, fn in _VARIANTS.items():
            ok = 0
            total = 0
            for _ssrc, _seq, _extended, raw in _captured:
                parsed = _parse_packet(raw)
                if parsed is None:
                    continue
                _ext, data = parsed
                try:
                    ct, aad, nonce = fn(raw, data)
                except ValueError:
                    continue
                total += 1
                try:
                    box.decrypt(ct, aad, nonce)
                    ok += 1
                except Exception:
                    pass
            print(f"[voice-debug] variant '{name}': ok={ok}/{total}")
            if ok > best_ok:
                best_name, best_fn, best_ok = name, fn, ok

        if best_fn is None:
            print("[voice-debug] no packets parseable")
            return

        if best_ok == 0:
            print("[voice-debug] ALL variants failed - raw structure samples printed above")
            return

        # Payload analysis using the winning variant
        proto_frames = 0
        passthrough_frames = 0
        frame_lens: dict[str, int] = {}
        for ssrc, seq, extended, raw in _captured:
            parsed = _parse_packet(raw)
            if parsed is None:
                continue
            ext, data = parsed
            try:
                ct, aad, nonce = best_fn(raw, data)
            except ValueError:
                continue
            try:
                plain = box.decrypt(ct, aad, nonce)
            except Exception:
                continue
            # plaintext = RTP ext body (8 bytes when present) + DAVE frame
            frame = plain[8:] if ext else plain
            if len(frame) >= 2:
                bucket = "<4" if len(frame) < 4 else "<16" if len(frame) < 16 else "<64" if len(frame) < 64 else ">=64"
                frame_lens[bucket] = frame_lens.get(bucket, 0) + 1
                if frame[-2:] == b"\xfa\xfa":
                    proto_frames += 1  # DAVE protocol frame (encrypted sender)
                else:
                    passthrough_frames += 1  # non-protocol frame (raw sender)
        print(
            f"[voice-debug] frames: protocol(0xFAFA)={proto_frames}, "
            f"non-protocol={passthrough_frames}"
        )
        print(f"[voice-debug] frame lengths: {frame_lens}")
        print(f"[voice-debug] winner: '{best_name}' ({best_ok} packets)")
