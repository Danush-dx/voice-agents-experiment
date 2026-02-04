"""
Vobiz WebSocket stream handler with complete voice agent loop.
Flow: User Audio -> VAD/Scribe -> LLM -> TTS -> User
"""
import json
import uuid
import asyncio
import base64
import audioop
import time
from fastapi import WebSocket

from src.config import config
from src.telephony.audio_converter import AudioConverter
from src.telephony.llm_service import LLMService
from src.telephony.tts_service import TTSService
from src.telephony.scribe_service import ScribeService
from src.telephony.groq_stt_service import GroqSTTService
from src.telephony.audio_player import AudioPlayer


class VobizStreamHandler:
    """Handles Vobiz WebSocket streams with real-time voice agent loop."""

    def __init__(self, vad_pipeline, asr_pipeline):
        """
        Initialize the handler.
        Note: vad_pipeline and asr_pipeline are kept for signature compatibility 
        but we use local VAD and configurable STT service.
        """
        self.vad_pipeline = vad_pipeline
        self.asr_pipeline = asr_pipeline

    async def handle_stream(self, websocket: WebSocket):
        """Handle Vobiz WebSocket stream."""
        stt_info = f"{config.GROQ_STT_MODEL} (Groq)" if config.ASR_TYPE == "groq" else f"{config.ELEVENLABS_STT_MODEL} (ElevenLabs)"
        print("=" * 60)
        print(f"VOICE AGENT STARTED ({stt_info}+{config.GROQ_LLM_MODEL}+{config.CARTESIA_TTS_MODEL})")
        print("=" * 60)
        
        try:
            await websocket.accept()
            print("✅ WebSocket ACCEPTED")
        except Exception as e:
            print(f"❌ WebSocket accept failed: {e}")
            return

        # Initialize services
        converter = AudioConverter()
        audio_player = AudioPlayer(websocket)
        
        try:
            llm_service = LLMService()
            tts_service = TTSService()
            
            # Select STT Service
            if config.ASR_TYPE == "groq":
                stt_service = GroqSTTService()
            else:
                stt_service = ScribeService()
            
            # Connect to STT service
            await stt_service.connect()
            
            print("✅ All services ready")
        except Exception as e:
            print(f"❌ Service init failed: {e}")
            return

        # VAD & Conversation State
        is_user_speaking = False
        silence_frames = 0
        SILENCE_THRESHOLD = 25  # ~500ms (20ms per frame)
        ENERGY_THRESHOLD = 1000  # RMS threshold
        
        packet_count = 0

        try:
            while True:
                message_text = await websocket.receive_text()
                message = json.loads(message_text)
                event = message.get("event")

                if event == "connected":
                    print("📡 Connected to Vobiz")

                elif event == "start":
                    print("🎙️ Stream STARTED")
                    # Initial Greeting
                    greeting = llm_service.get_greeting()
                    print(f"🤖 Greeting: {greeting}")
                    
                    # Stream greeting
                    tts_stream = tts_service.synthesize_stream(greeting)
                    if tts_stream:
                        await audio_player.stream_audio(tts_stream)

                elif event == "media":
                    packet_count += 1
                    media = message.get("media", {})
                    payload = media.get("payload")
                    if not payload:
                        continue
                    
                    # 1. Convert Audio (µ-law -> PCM)
                    pcm = converter.convert(payload)
                    
                    # 2. Feed STT service
                    await stt_service.send_audio(pcm)
                    
                    # 3. VAD / Barge-in Logic
                    energy = audioop.rms(pcm, 2)
                    
                    if energy > ENERGY_THRESHOLD:
                        if not is_user_speaking:
                            print(f"🎤 User started speaking (Energy: {energy})")
                            is_user_speaking = True
                            
                            # BARGE-IN: Stop agent if speaking
                            if audio_player.is_playing:
                                audio_player.stop()
                        
                        silence_frames = 0
                        
                    elif is_user_speaking:
                        silence_frames += 1
                        
                        # End of turn detection
                        if silence_frames >= SILENCE_THRESHOLD:
                            print("🔇 Silence detected - End of turn")
                            is_user_speaking = False
                            silence_frames = 0
                            
                            rtt_start = time.perf_counter()
                            
                            # 1. Get transcript from STT
                            stt_start = time.perf_counter()
                            user_text = await stt_service.get_and_clear_transcript()
                            stt_latency = (time.perf_counter() - stt_start) * 1000
                            
                            if user_text:
                                print(f"📝 User Said: '{user_text}'")
                                
                                # 2. Generate Response (LLM)
                                llm_start = time.perf_counter()
                                response = await llm_service.generate_response(user_text)
                                llm_latency = (time.perf_counter() - llm_start) * 1000
                                print(f"🤖 Agent Responding: '{response}'")
                                
                                # 3. Stream Synthesis (TTS)
                                tts_start = time.perf_counter()
                                tts_stream = tts_service.synthesize_stream(response)
                                tts_latency = (time.perf_counter() - tts_start) * 1000
                                
                                # 4. Calculate Initial Latency (TTFB)
                                rtt_total = (time.perf_counter() - rtt_start) * 1000
                                
                                print(f"⏱️  LATENCY METRICS:")
                                print(f"   - STT Delay: {stt_latency:.2f}ms")
                                print(f"   - LLM Time:  {llm_latency:.2f}ms")
                                print(f"   - TTS Init:  {tts_latency:.2f}ms")
                                print(f"   - Total TTFB: {rtt_total:.2f}ms")
                                
                                # Stream Audio
                                if tts_stream:
                                    await audio_player.stream_audio(tts_stream)
                            else:
                                print("⚠️ No transcript received yet")

                elif event == "stop":
                    print("🛑 Stream stopped")
                    break

        except Exception as e:
            if "1000" not in str(e) and "1001" not in str(e):
                print(f"❌ Error in handle_stream: {e}")
                import traceback
                traceback.print_exc()

        finally:
            print("📴 Call ended")
            await stt_service.close()
            llm_service.reset_conversation()
            audio_player.stop()
