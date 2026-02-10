"""
Groq STT Service using Whisper on Groq.
Provides fast transcription for voice agent.
"""
import io
import wave
import asyncio
import logging
from groq import AsyncGroq
from src.config import config

logger = logging.getLogger(__name__)

# ~5 minutes of 16kHz 16-bit mono PCM
MAX_BUFFER_SIZE = 10_000_000


class GroqSTTService:
    """Service for speech-to-text using Groq's Whisper API."""

    def __init__(self):
        """Initialize Groq client."""
        api_key = config.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable required")

        logger.info(f"GroqSTTService: Initializing with model: {config.GROQ_STT_MODEL}")
        self.client = AsyncGroq(api_key=api_key)
        self.model = config.GROQ_STT_MODEL
        self.buffer = io.BytesIO()
        self.is_connected = False

    async def connect(self):
        """Mock connect for interface compatibility with ScribeService."""
        self.is_connected = True
        logger.info("GroqSTTService: Ready (Buffered mode)")
        return True

    async def send_audio(self, pcm_bytes: bytes):
        """Accumulate PCM audio chunks in buffer."""
        if not self.is_connected:
            return
        if self.buffer.tell() > MAX_BUFFER_SIZE:
            logger.warning("STT buffer overflow, clearing")
            self.buffer = io.BytesIO()
        self.buffer.write(pcm_bytes)

    async def get_and_clear_transcript(self) -> str:
        """
        Synthesize accumulated audio and clear buffer.

        Returns:
            Transcribed text
        """
        audio_size = self.buffer.tell()
        if audio_size == 0:
            return ""

        # Prepare WAV file in memory
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(16000)
            wav_file.writeframes(self.buffer.getvalue())

        wav_io.seek(0)
        wav_io.name = "audio.wav"

        try:
            logger.info(f"GroqSTTService: Transcribing {audio_size} bytes...")

            # Context prompt to guide the model towards Indian languages and Romanized script
            prompt_text = (
                "English, Hindi, Tamil, Kannada, Telugu. "
                "Namaste, Vanakkam, Namaskaram. "
                "Kya haal hai? Tamil theriyuma? Bagunnara? Chennagidira?"
            )

            transcription = await self.client.audio.transcriptions.create(
                file=wav_io,
                model=self.model,
                prompt=prompt_text,
                response_format="json",
            )

            text = transcription.text.strip()
            return text

        except Exception as e:
            logger.error(f"GroqSTTService error: {e}")
            return ""
        finally:
            # Clear buffer for next turn
            self.buffer = io.BytesIO()

    def clear_buffer(self):
        """Discard all buffered audio (echo cleanup)."""
        self.buffer = io.BytesIO()
        logger.debug("GroqSTTService: Buffer cleared")

    async def close(self):
        """Close service."""
        self.is_connected = False
        self.buffer = io.BytesIO()
        await self.client.close()
        logger.info("GroqSTTService: Closed")
