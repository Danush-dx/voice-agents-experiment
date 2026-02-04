"""
ElevenLabs Scribe v2 Realtime Service.
Handles WebSocket connection for real-time speech-to-text.
"""
import os
import asyncio
import base64
import json
from elevenlabs import ElevenLabs, RealtimeEvents
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
        # We don't need a manual receiver task if using 'on' handlers, 
        # but we might need to keep the connection alive or processing. 
        # The SDK likely runs a background task for the websocket.

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
            
            # Register Event Handlers
            self.connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, self._on_transcript)
            self.connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, self._on_transcript)
            self.connection.on(RealtimeEvents.ERROR, self._on_error)
            self.connection.on(RealtimeEvents.CLOSE, self._on_close)
            
            self.is_connected = True
            print("✅ ScribeService: Connected")
            
        except Exception as e:
            print(f"❌ ScribeService connection error: {e}")
            self.is_connected = False
            raise

    def _on_transcript(self, event):
        """Handle transcript events."""
        # Check event structure. It's likely an object with 'text'.
        try:
            text = ""
            if hasattr(event, 'text'):
                text = event.text
            elif isinstance(event, dict):
                text = event.get('text', '')
            
            if text:
                print(f"📝 Scribe: {text}")
                self.current_transcript.append(text)
        except Exception as e:
            print(f"❌ Error processing transcript event: {e}")

    def _on_error(self, event):
        print(f"❌ Scribe Error Event: {event}")

    def _on_close(self, event=None):
        print(f"Scribe connection closed. Event: {event}")
        self.is_connected = False

    async def send_audio(self, pcm_bytes: bytes):
        """Send PCM audio chunk to Scribe."""
        if not self.is_connected or not self.connection:
            return
        
        try:
            # Base64 encode
            b64_audio = base64.b64encode(pcm_bytes).decode("utf-8")
            # Send as JSON payload per help docs
            await self.connection.send({"audio_base_64": b64_audio})
        except Exception as e:
            print(f"❌ ScribeService send error: {e}")

    def get_and_clear_transcript(self) -> str:
        """Return accumulated transcript and clear buffer."""
        # Join with space and clear
        full_text = " ".join(self.current_transcript).strip()
        self.current_transcript = []
        return full_text

    async def close(self):
        """Close connection."""
        if self.connection:
            try:
                await self.connection.close()
            except:
                pass
        self.is_connected = False
