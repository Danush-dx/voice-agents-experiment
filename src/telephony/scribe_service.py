"""
ElevenLabs Scribe v2 Realtime Service.
Handles WebSocket connection for real-time speech-to-text.
"""
import os
import asyncio
import base64
import json
from elevenlabs.client import ElevenLabs
from elevenlabs.realtime import RealtimeAudioOptions, AudioFormat
from src.config import config

class ScribeService:
    """Service for real-time transcription using ElevenLabs Scribe v2."""
    
    def __init__(self):
        """Initialize Scribe service."""
        api_key = config.ELEVENLABS_API_KEY
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable required")
        
        self.client = ElevenLabs(api_key=api_key)
        self.connection = None
        self.current_transcript = []
        self.is_connected = False
        self.receiver_task = None

    async def connect(self):
        """Establish WebSocket connection to Scribe."""
        try:
            print(f"🔧 ScribeService: Connecting with model: {config.ELEVENLABS_STT_MODEL}")
            
            # Use RealtimeAudioOptions for manual audio streaming (PCM 16kHz)
            options = RealtimeAudioOptions(
                model_id=config.ELEVENLABS_STT_MODEL,
                audio_format=AudioFormat.PCM_16000,
                sample_rate=16000
            )
            
            self.connection = await self.client.speech_to_text.realtime.connect(options)
            self.is_connected = True
            
            # Start background receiver
            self.receiver_task = asyncio.create_task(self._receive_loop())
            print("✅ ScribeService: Connected")
            
        except Exception as e:
            print(f"❌ ScribeService connection error: {e}")
            self.is_connected = False
            raise

    async def _receive_loop(self):
        """Background loop to receive transcripts."""
        try:
            async for event in self.connection:
                # Scribe v2 realtime events often have 'text' attribute
                if hasattr(event, 'text') and event.text:
                    text = event.text.strip()
                    if text:
                        self.current_transcript.append(text)
                        print(f"📝 Scribe Partial: {text}")
                elif isinstance(event, dict) and 'text' in event:
                     text = event['text'].strip()
                     if text:
                        self.current_transcript.append(text)
                        print(f"📝 Scribe Partial (dict): {text}")
                        
        except Exception as e:
            # Silently handle task cancellation or closed connections
            if self.is_connected:
                print(f"❌ ScribeService receive error: {e}")
        finally:
            self.is_connected = False

    async def send_audio(self, pcm_bytes: bytes):
        """Send PCM audio chunk to Scribe."""
        if not self.is_connected or not self.connection:
            return
        
        try:
            # Send audio chunk
            await self.connection.send_audio(pcm_bytes)
        except Exception as e:
            print(f"❌ ScribeService send error: {e}")

    def get_and_clear_transcript(self) -> str:
        """Return accumulated transcript and clear buffer."""
        full_text = " ".join(self.current_transcript).strip()
        self.current_transcript = []
        return full_text

    async def close(self):
        """Close connection."""
        self.is_connected = False
        if self.receiver_task:
            self.receiver_task.cancel()
        if self.connection:
            try:
                # The SDK connection might not have an explicit close, 
                # but canceling the receiver loop usually suffices.
                pass
            except:
                pass