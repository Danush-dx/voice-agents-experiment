"""
Audio converter for Vobiz telephony integration.
Converts 8kHz µ-law audio to 16kHz PCM format.
"""
import base64
import audioop


class AudioConverter:
    """Converts Vobiz audio format (8kHz µ-law) to VoiceStreamAI format (16kHz PCM)."""

    def __init__(self):
        """Initialize the audio converter with empty resample state."""
        self.resample_state = None

    def convert(self, base64_mulaw: str) -> bytes:
        """
        Convert base64-encoded µ-law audio to 16kHz PCM.

        Args:
            base64_mulaw: Base64-encoded µ-law audio data

        Returns:
            16kHz 16-bit PCM audio bytes
        """
        # Step 1: Decode base64
        mulaw_bytes = base64.b64decode(base64_mulaw)

        # Step 2: Convert µ-law to PCM (16-bit)
        pcm_8khz = audioop.ulaw2lin(mulaw_bytes, 2)

        # Step 3: Resample from 8kHz to 16kHz
        pcm_16khz, self.resample_state = audioop.ratecv(
            pcm_8khz,
            2,          # sample width (16-bit)
            1,          # channels (mono)
            8000,       # input rate
            16000,      # output rate
            self.resample_state
        )

        return pcm_16khz

    def reset(self):
        """Reset the converter state for a new call."""
        self.resample_state = None
