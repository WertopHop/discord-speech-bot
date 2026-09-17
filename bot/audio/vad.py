"""Voice Activity Detector (VAD).

The main engine is webrtcvad, the built-in one is used
energy VAD with an adaptive noise threshold - it is noticeably simpler, but for
Voice chat in a quiet room works acceptably.
"""
from __future__ import annotations

import numpy as np

from .pcm import VAD_SAMPLE_RATE

try:
    import webrtcvad

    _HAVE_WEBRTC = True
except ImportError:
    webrtcvad = None
    _HAVE_WEBRTC = False

# webrtcvad accepts frames strictly 10/20/30 ms; take 20 ms (320 samples at 16 kHz)
FRAME_20MS = VAD_SAMPLE_RATE * 20 // 1000


class VoiceActivityDetector:
    """Classifier of 20 ms mono-16kHz frames: speech/silence."""

    def __init__(self, aggressiveness: int = 2) -> None:
        self._vad = webrtcvad.Vad(int(aggressiveness)) if _HAVE_WEBRTC else None
        self._noise_floor = 250.0

    @property
    def backend(self) -> str:
        return "webrtcvad" if self._vad is not None else "energy"

    @property
    def frame_samples(self) -> int:
        """Samples per classification frame (20 ms at 16 kHz)."""
        return FRAME_20MS

    def process(self, mono16k: np.ndarray) -> list[bool]:
        """Splits the input into 20ms frames and returns a list of 'speech?' for each."""
        count = len(mono16k) // FRAME_20MS
        return [
            self._classify(mono16k[i * FRAME_20MS : (i + 1) * FRAME_20MS])
            for i in range(count)
        ]

    def _classify(self, chunk: np.ndarray) -> bool:
        if self._vad is not None:
            return self._vad.is_speech(chunk.tobytes(), VAD_SAMPLE_RATE)
        if len(chunk) == 0:
            return False
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        threshold = max(500.0, self._noise_floor * 3.0)
        speech = rms > threshold
        if not speech:
            # Slowly increase the noise rating down/up from silent frames
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
        return speech
