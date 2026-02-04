"""
Vobiz WebSocket stream handler with complete voice agent loop.
Flow: TTS greeting → USER speaks → STT → LLM → TTS → USER → (repeat)
"""
import json
import uuid
import asyncio
import base64
from fastapi import WebSocket

from src.client import Client
from src.telephony.audio_converter import AudioConverter
from src.telephony.llm_service import LLMService
from src.telephony.tts_service import TTSService


class VobizStreamHandler:
    """Handles Vobiz WebSocket streams with voice agent loop."""

    def __init__(self, vad_pipeline, asr_pipeline):
        """Initialize the handler."""
        self.vad_pipeline = vad_pipeline
        self.asr_pipeline = asr_pipeline

    async def handle_stream(self, websocket: WebSocket):
        """Handle Vobiz WebSocket stream with full conversation loop."""
        print("=" * 60)
        print("VOICE AGENT STARTED")
        print("=" * 60)
        
        try:
            await websocket.accept()
            print("✅ WebSocket ACCEPTED")
        except Exception as e:
            print(f"❌ WebSocket accept failed: {e}")
            return

        # Initialize services
        converter = AudioConverter()
        
        try:
            llm_service = LLMService()
            tts_service = TTSService()
            print("✅ LLM and TTS services ready")
        except Exception as e:
            print(f"❌ Service init failed: {e}")
            return

        # Conversation state
        audio_buffer = bytearray()
        is_speaking = False
        silence_frames = 0
        SILENCE_THRESHOLD = 25  # ~500ms

        try:
            while True:
                message_text = await websocket.receive_text()
                message = json.loads(message_text)
                event = message.get("event")

                if event == "connected":
                    print("📡 Connected to Vobiz")

                elif event == "start":
                    print("🎙️ Stream STARTED")
                    
                    # Play greeting - fire and forget
                    greeting = llm_service.get_greeting()
                    audio = tts_service.synthesize(greeting)
                    if audio:
                        asyncio.create_task(self._send_audio(websocket, audio))
                        print("✅ Greeting queued")

                elif event == "media":
                    media = message.get("media", {})
                    payload = media.get("payload")
                    if not payload:
                        continue
                    
                    # Convert audio
                    pcm = converter.convert(payload)
                    
                    # Energy-based VAD
                    energy = sum(abs(b - 128) for b in pcm[:100]) / 100
                    
                    if energy > 8:
                        is_speaking = True
                        silence_frames = 0
                        audio_buffer.extend(pcm)
                    elif is_speaking:
                        silence_frames += 1
                        audio_buffer.extend(pcm)
                        
                        if silence_frames >= SILENCE_THRESHOLD:
                            if len(audio_buffer) > 16000:
                                print(f"🎤 Processing {len(audio_buffer)} bytes...")
                                
                                # Transcribe
                                text = await self._transcribe(bytes(audio_buffer))
                                if text and text.strip():
                                    print(f"📝 User: '{text}'")
                                    
                                    # LLM
                                    response = llm_service.generate_response(text)
                                    print(f"🤖 Agent: '{response}'")
                                    
                                    # TTS - fire and forget
                                    audio = tts_service.synthesize(response)
                                    if audio:
                                        asyncio.create_task(self._send_audio(websocket, audio))
                            
                            audio_buffer.clear()
                            is_speaking = False
                            silence_frames = 0

                elif event == "stop":
                    print("🛑 Stream stopped")
                    break

        except Exception as e:
            # 1000 is normal close, ignore it
            if "1000" not in str(e):
                print(f"❌ Error: {e}")

        finally:
            print("📴 Call ended")
            llm_service.reset_conversation()

    async def _send_audio(self, ws: WebSocket, audio: bytes):
        """Send audio to Vobiz without blocking."""
        try:
            chunk_size = 320
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                await ws.send_text(json.dumps({
                    "event": "playAudio",
                    "media": {
                        "contentType": "audio/x-l16",
                        "sampleRate": 8000,
                        "payload": base64.b64encode(chunk).decode()
                    }
                }))
            print(f"✅ Sent {len(audio)} bytes")
        except Exception as e:
            print(f"❌ Send error: {e}")

    async def _transcribe(self, audio: bytes) -> str:
        """Transcribe audio."""
        try:
            client = Client(client_id="temp", sampling_rate=16000, samples_width=2)
            client.scratch_buffer = bytearray(audio)
            result = await self.asr_pipeline.transcribe(client)
            return result.get("text", "")
        except Exception as e:
            print(f"❌ ASR error: {e}")
            return ""
