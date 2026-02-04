"""
Audio player for sending audio back to Vobiz via WebSocket.
Sends PCM audio in playAudio events.
"""
import json
import base64
import asyncio
from fastapi import WebSocket


class AudioPlayer:
    """Handles sending audio back to Vobiz caller."""
    
    def __init__(self, websocket: WebSocket):
        """Initialize audio player."""
        self.websocket = websocket
        self.sample_rate = 8000
        # 20ms of 8kHz 16-bit mono PCM = 320 bytes
        self.chunk_size = 320
    
    async def play_audio(self, audio_bytes: bytes):
        """
        Send audio to caller via Vobiz playAudio event.
        
        Args:
            audio_bytes: Raw PCM audio (8kHz, 16-bit, mono)
        """
        if not audio_bytes:
            print("⚠️ AudioPlayer: No audio bytes to play")
            return
        
        print(f"🔊 AudioPlayer: Sending {len(audio_bytes)} PCM bytes as audio/x-l16...")
        
        chunk_count = 0
        try:
            for i in range(0, len(audio_bytes), self.chunk_size):
                chunk = audio_bytes[i:i + self.chunk_size]
                
                # Create playAudio event - using PCM format (audio/x-l16)
                event = {
                    "event": "playAudio",
                    "media": {
                        "contentType": "audio/x-l16",
                        "sampleRate": self.sample_rate,
                        "payload": base64.b64encode(chunk).decode("utf-8")
                    }
                }
                
                await self.websocket.send_text(json.dumps(event))
                chunk_count += 1
                
                # Pace to real-time (20ms per chunk)
                await asyncio.sleep(0.020)
            
            print(f"✅ AudioPlayer: Sent {chunk_count} PCM chunks")
            
        except Exception as e:
            print(f"❌ AudioPlayer error: {e}")
            import traceback
            traceback.print_exc()
