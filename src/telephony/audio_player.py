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

    async def stream_audio(self, audio_generator):
        """
        Stream audio chunks from a generator to the caller.
        
        Args:
            audio_generator: Iterator/Generator yielding PCM audio bytes
        """
        # Stop any current playback
        self.stop()
        
        # Create new playback task
        self.is_playing = True
        self.current_task = asyncio.create_task(self._consume_stream(audio_generator))
        
    def stop(self):
        """Stop current playback immediately."""
        if self.is_playing:
            print("🛑 AudioPlayer: Stopping playback (Barge-in)")
            self.is_playing = False
        
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            self.current_task = None

    async def _consume_stream(self, audio_generator):
        """Consume audio generator and stream chunks with buffering."""
        buffer = bytearray()
        try:
            for chunk in audio_generator:
                if not self.is_playing:
                    break
                
                buffer.extend(chunk)
                
                # Process complete chunks from buffer
                while len(buffer) >= self.chunk_size:
                    if not self.is_playing:
                        break
                        
                    # Extract one chunk
                    chunk_to_send = buffer[:self.chunk_size]
                    del buffer[:self.chunk_size]
                    
                    await self._send_single_chunk(chunk_to_send)
            
            # Process remaining bytes if they form a complete sample (multiple of 2)
            if self.is_playing and len(buffer) > 0:
                # Ensure we have an even number of bytes for 16-bit PCM
                if len(buffer) % 2 != 0:
                    buffer = buffer[:-1]  # Drop last byte if incomplete sample
                
                if len(buffer) > 0:
                    await self._send_single_chunk(buffer)
            
            if self.is_playing:
                print("✅ AudioPlayer: Finished streaming")
        except asyncio.CancelledError:
            print("🛑 AudioPlayer: Stream Task Cancelled")
        except Exception as e:
            print(f"❌ AudioPlayer stream error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_playing = False

    async def _send_single_chunk(self, chunk: bytes):
        """Convert and send a single PCM chunk."""
        try:
            # Convert 16-bit Linear PCM to 8-bit µ-law
            mulaw_chunk = audioop.lin2ulaw(chunk, 2)
            
            # Create playAudio event
            event = {
                "event": "playAudio",
                "media": {
                    "contentType": "audio/x-mulaw",
                    "sampleRate": self.sample_rate,
                    "payload": base64.b64encode(mulaw_chunk).decode("utf-8")
                }
            }
            
            await self.websocket.send_text(json.dumps(event))
            
            # Pace to real-time (20ms per chunk is ideal, slightly faster to avoid underrun)
            # 320 bytes / 2 bytes/sample / 8000 samples/sec = 0.02s = 20ms
            await asyncio.sleep(0.018) 
        except Exception as e:
            print(f"❌ Error sending chunk: {e}")
            raise e

    async def _stream_audio_chunk(self, audio_bytes: bytes):
        """Deprecated: Use _consume_stream buffering logic instead."""
        pass 

    async def _stream_audio(self, audio_bytes: bytes):
        """Internal method to stream audio chunks."""
        if not audio_bytes:
            return
        
        print(f"🔊 AudioPlayer: Converting {len(audio_bytes)} PCM bytes to µ-law and sending...")
        
        # Reuse the buffering logic by simulating a generator
        async def byte_generator():
            yield audio_bytes
            
        await self._consume_stream(byte_generator())
