"""
TTS Service using Cartesia API.
Provides text-to-speech using sonic model.
"""
import os
from cartesia import Cartesia
from src.config import config


class TTSService:
    """Cartesia TTS service for voice agent speech synthesis."""
    
    def __init__(self):
        """Initialize Cartesia client."""
        api_key = config.CARTESIA_API_KEY
        if not api_key:
            raise ValueError("CARTESIA_API_KEY environment variable required")
        
        print(f"🔧 TTSService: Initializing Cartesia with model: {config.CARTESIA_TTS_MODEL}")
        self.client = Cartesia(api_key=api_key)
        self.model_id = config.CARTESIA_TTS_MODEL
        self.voice_id = config.CARTESIA_VOICE_ID
        print(f"✅ TTSService: Initialized with model={self.model_id}, voice={self.voice_id}")
    
    def synthesize(self, text: str) -> bytes:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Raw PCM audio bytes (8kHz, 16-bit, mono)
        """
        if not text or not text.strip():
            print("⚠️ TTSService: Empty text, returning no audio")
            return b""
            
        try:
            print(f"🔊 TTSService: Synthesizing: '{text}'")
            
            # Use Cartesia SDK - returns generator, consume it
            audio_generator = self.client.tts.bytes(
                model_id=self.model_id,
                transcript=text,
                voice={"mode": "id", "id": self.voice_id},
                output_format={
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": 8000
                }
            )
            
            # Consume generator and concatenate all chunks
            audio_data = b"".join(audio_generator)
            
            print(f"✅ TTSService: Generated {len(audio_data)} bytes of audio")
            return audio_data
            
        except Exception as e:
            print(f"❌ TTSService error: {e}")
            import traceback
            traceback.print_exc()
            return b""

    def synthesize_stream(self, text: str, language: str = "en"):
        """
        Synthesize speech from text and yield audio chunks.
        
        Args:
            text: Text to convert to speech
            language: ISO language code (en, hi, ta, te, kn)
            
        Yields:
            Raw PCM audio bytes (8kHz, 16-bit, mono)
        """
        if not text or not text.strip():
            print("⚠️ TTSService: Empty text, returning no audio")
            return
            
        try:
            print(f"🔊 TTSService: Streaming synthesis for: '{text}' (Lang: {language})")
            
            # Use Cartesia SDK - returns generator
            audio_generator = self.client.tts.bytes(
                model_id=self.model_id,
                transcript=text,
                voice={"mode": "id", "id": self.voice_id},
                language=language,
                output_format={
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": 8000
                }
            )
            
            chunk_count = 0
            for chunk in audio_generator:
                if chunk:
                    chunk_count += 1
                    yield chunk
            
            print(f"✅ TTSService: Streaming complete ({chunk_count} chunks)")
            
        except Exception as e:
            print(f"❌ TTSService error during streaming: {e}")
            import traceback
            traceback.print_exc()
