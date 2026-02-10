"""
Silero VAD for real-time telephony.
Frame-by-frame speech probability using Silero's pre-trained model.
"""
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SileroVAD:
    """Frame-by-frame Silero VAD for real-time telephony."""

    def __init__(self, threshold=0.5, sample_rate=16000, chunk_samples=512):
        """
        Initialize Silero VAD.

        Args:
            threshold: Speech probability threshold (default 0.5)
            sample_rate: Audio sample rate in Hz (default 16000)
            chunk_samples: Samples per chunk (512 = 32ms at 16kHz, optimal for Silero)
        """
        self.model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples  # 512 samples = 32ms at 16kHz
        self._buffer = bytes()
        logger.info(f"SileroVAD initialized: threshold={threshold}, chunk={chunk_samples} samples")

    def process_chunk(self, pcm_bytes: bytes) -> float:
        """
        Feed 16kHz 16-bit PCM, returns speech probability 0.0-1.0.

        Args:
            pcm_bytes: Raw PCM audio bytes (16kHz, 16-bit, mono)

        Returns:
            Speech probability between 0.0 and 1.0
        """
        self._buffer += pcm_bytes
        bytes_per_chunk = self.chunk_samples * 2  # 16-bit = 2 bytes/sample

        if len(self._buffer) < bytes_per_chunk:
            return 0.0  # Not enough data yet

        # Take exactly one chunk
        chunk = self._buffer[:bytes_per_chunk]
        self._buffer = self._buffer[bytes_per_chunk:]

        # Convert to float32 tensor
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)

        prob = self.model(tensor, self.sample_rate).item()
        return prob

    def reset_states(self):
        """Reset model states and internal buffer. Call at call boundaries."""
        self.model.reset_states()
        self._buffer = bytes()
