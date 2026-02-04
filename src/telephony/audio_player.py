"""
Audio player for sending audio back to Vobiz via WebSocket.
Sends PCM audio in playAudio events.
"""
import json
import base64
import asyncio
import audioop
from fastapi import WebSocket


class AudioPlayer:
    """Handles sending audio back to Vobiz caller with interruption support."""
    
    def __init__(self, websocket: WebSocket):
        """Initialize audio player."""
        self.websocket = websocket
        self.sample_rate = 8000
        # 20ms of 8kHz 16-bit mono PCM = 320 bytes
        self.chunk_size = 320
        self.current_task = None
        self.is_playing = False
    
    async def play_audio(self, audio_bytes: bytes):
        """
        Send audio to caller via Vobiz playAudio event.
        Cancels any existing playback.
        
        Args:
            audio_bytes: Raw PCM audio (8kHz, 16-bit, mono)
        """
        # Stop any current playback
        self.stop()
        
        # Create new playback task
        self.is_playing = True
        self.current_task = asyncio.create_task(self._stream_audio(audio_bytes))
        
    def stop(self):
        """Stop current playback immediately."""
        if self.is_playing:
            print("🛑 AudioPlayer: Stopping playback (Barge-in)")
            self.is_playing = False
        
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            self.current_task = None

    async def _stream_audio(self, audio_bytes: bytes):
        """Internal method to stream audio chunks."""
        if not audio_bytes:
            return
        
        print(f"🔊 AudioPlayer: Converting {len(audio_bytes)} PCM bytes to µ-law and sending...")
        
        chunk_count = 0
        try:
            for i in range(0, len(audio_bytes), self.chunk_size):
                if not self.is_playing:
                    break
                    
                chunk = audio_bytes[i:i + self.chunk_size]
                
                # Convert 16-bit Linear PCM to 8-bit µ-law
                mulaw_chunk = audioop.lin2ulaw(chunk, 2)
                
                # Create playAudio event - using µ-law format
                event = {
                    "event": "playAudio",
                    "media": {
                        "contentType": "audio/x-mulaw",
                        "sampleRate": self.sample_rate,
                        "payload": base64.b64encode(mulaw_chunk).decode("utf-8")
                    }
                }
                
                await self.websocket.send_text(json.dumps(event))
                chunk_count += 1
                
                # Pace to real-time (20ms per chunk)
                await asyncio.sleep(0.018) # Slightly faster than realtime to prevent buffer underrun
            
            if self.is_playing:
                print(f"✅ AudioPlayer: Finished sending {chunk_count} chunks")
            else:
                print(f"🛑 AudioPlayer: Interrupted after {chunk_count} chunks")
            
        except asyncio.CancelledError:
            print("🛑 AudioPlayer: Task Cancelled")
        except Exception as e:
            print(f"❌ AudioPlayer error: {e}")
        finally:
            self.is_playing = False
