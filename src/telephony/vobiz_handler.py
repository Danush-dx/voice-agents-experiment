"""
Vobiz WebSocket stream handler with complete voice agent loop.
Flow: User Audio -> Silero VAD -> STT (echo-gated) -> LLM streaming -> TTS per-sentence -> User
"""
import json
import re
import asyncio
import time
import logging
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from src.config import config
from src.telephony.audio_converter import AudioConverter
from src.telephony.llm_service import LLMService
from src.telephony.tts_service import TTSService
from src.telephony.scribe_service import ScribeService
from src.telephony.groq_stt_service import GroqSTTService
from src.telephony.audio_player import AudioPlayer
from src.vad.silero_vad import SileroVAD

logger = logging.getLogger(__name__)

# Silero VAD thresholds (industry standard with hysteresis)
SPEECH_ONSET_THRESHOLD = 0.5
SPEECH_OFFSET_THRESHOLD = 0.3
SILENCE_THRESHOLD = 25  # ~500ms for end-of-turn

# Barge-in requires sustained speech (avoid false triggers from echo)
BARGE_IN_FRAMES_REQUIRED = 3  # ~96ms at 32ms/frame

# Max base64 payload size (~75KB decoded)
MAX_PAYLOAD_SIZE = 100_000

# Sentence boundary: punctuation followed by space or end-of-string
SENTENCE_END_RE = re.compile(r'[.?!]+(?:\s|$)')


class VobizStreamHandler:
    """Handles Vobiz WebSocket streams with real-time voice agent loop."""

    def __init__(self):
        """Initialize the handler. Loads Silero VAD once for reuse across connections."""
        self.silero_vad = SileroVAD()

    async def handle_stream(self, websocket: WebSocket):
        """Handle Vobiz WebSocket stream."""
        stt_info = f"{config.GROQ_STT_MODEL} (Groq)" if config.ASR_TYPE == "groq" else f"{config.ELEVENLABS_STT_MODEL} (ElevenLabs)"
        logger.info(f"VOICE AGENT STARTED ({stt_info}+{config.GROQ_LLM_MODEL}+{config.CARTESIA_TTS_MODEL})")

        try:
            await websocket.accept()
            logger.info("WebSocket ACCEPTED")
        except Exception as e:
            logger.error(f"WebSocket accept failed: {e}")
            return

        # Initialize services
        converter = AudioConverter()
        audio_player = AudioPlayer(websocket)
        self.silero_vad.reset_states()

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

            logger.info("All services ready")
        except Exception as e:
            logger.error(f"Service init failed: {e}")
            return

        # VAD & Conversation State
        is_user_speaking = False
        silence_frames = 0
        barge_in_frames = 0  # Consecutive high-prob frames during playback

        packet_count = 0
        current_turn_task = None

        # Hallucination phrases
        HALLUCINATIONS = [
            "vielen dank", "thank you for watching", "sous-titres",
            "amara.org", "mbc", "subtitles"
        ]

        async def llm_tts_pipeline(text, stt_latency, rtt_start):
            """
            Async generator: streams LLM tokens, buffers into sentences,
            synthesizes each sentence via TTS, and yields audio chunks.
            """
            token_buffer = ""
            first_sentence = True

            async for token in llm_service.generate_stream(text, language="en"):
                token_buffer += token

                # Check for sentence boundary
                match = SENTENCE_END_RE.search(token_buffer)
                while match:
                    # Split at sentence boundary
                    end_pos = match.end()
                    sentence = token_buffer[:end_pos].strip()
                    token_buffer = token_buffer[end_pos:]

                    if sentence:
                        if first_sentence:
                            ttfb = (time.perf_counter() - rtt_start) * 1000
                            logger.info(
                                f"LATENCY: STT={stt_latency:.0f}ms "
                                f"TTFB={ttfb:.0f}ms (first sentence: '{sentence}')"
                            )
                            first_sentence = False

                        # Clear STT buffer before playing (prevent echo transcription)
                        stt_service.clear_buffer()

                        # Synthesize sentence and yield audio chunks
                        tts_gen = tts_service.synthesize_stream(sentence, language="en")
                        if tts_gen:
                            for audio_chunk in tts_gen:
                                yield audio_chunk

                    # Check for more sentence boundaries in remaining buffer
                    match = SENTENCE_END_RE.search(token_buffer)

            # Flush remaining text
            remaining = token_buffer.strip()
            if remaining:
                if first_sentence:
                    ttfb = (time.perf_counter() - rtt_start) * 1000
                    logger.info(
                        f"LATENCY: STT={stt_latency:.0f}ms "
                        f"TTFB={ttfb:.0f}ms (first sentence: '{remaining}')"
                    )

                stt_service.clear_buffer()
                tts_gen = tts_service.synthesize_stream(remaining, language="en")
                if tts_gen:
                    for audio_chunk in tts_gen:
                        yield audio_chunk

        async def process_turn(text, stt_latency, rtt_start):
            """Handle the streaming LLM→TTS pipeline in background."""
            try:
                logger.info(f"Agent processing: '{text}'")
                pipeline = llm_tts_pipeline(text, stt_latency, rtt_start)
                await audio_player.stream_audio_async(pipeline)
            except asyncio.CancelledError:
                logger.info("Turn processing cancelled (barge-in)")
            except Exception as e:
                logger.error(f"Error in process_turn: {e}", exc_info=True)

        try:
            while True:
                message_text = await websocket.receive_text()
                message = json.loads(message_text)
                event = message.get("event")

                if event == "connected":
                    logger.info("Connected to Vobiz")

                elif event == "start":
                    logger.info("Stream STARTED")
                    greeting = llm_service.get_greeting()
                    logger.debug(f"Greeting: {greeting}")
                    tts_stream = tts_service.synthesize_stream(greeting, language="en")
                    if tts_stream:
                        await audio_player.stream_audio(tts_stream)

                elif event == "media":
                    packet_count += 1
                    media = message.get("media", {})
                    payload = media.get("payload")
                    if not payload or len(payload) > MAX_PAYLOAD_SIZE:
                        continue

                    try:
                        pcm = converter.convert(payload)
                    except Exception:
                        continue

                    # Silero VAD (always run, even during playback for barge-in detection)
                    speech_prob = self.silero_vad.process_chunk(pcm)

                    # --- ECHO GATE ---
                    if audio_player.is_playing:
                        # Do NOT feed audio to STT during playback (would transcribe echo)
                        # Only monitor VAD for barge-in
                        if speech_prob > SPEECH_ONSET_THRESHOLD:
                            barge_in_frames += 1
                            if barge_in_frames >= BARGE_IN_FRAMES_REQUIRED:
                                logger.info(f"BARGE-IN detected ({barge_in_frames} frames, prob: {speech_prob:.2f})")
                                # Stop playback
                                audio_player.stop()
                                # Cancel active LLM/TTS task
                                if current_turn_task and not current_turn_task.done():
                                    current_turn_task.cancel()
                                    try:
                                        await asyncio.wait_for(current_turn_task, timeout=0.1)
                                    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                                        pass
                                # Clear STT buffer (discard echo audio)
                                stt_service.clear_buffer()
                                # Start new turn
                                is_user_speaking = True
                                silence_frames = 0
                                barge_in_frames = 0
                                # Feed this frame to STT (first real user audio)
                                await stt_service.send_audio(pcm)
                        else:
                            barge_in_frames = 0
                        continue  # Skip normal VAD/STT processing during playback

                    # --- NORMAL MODE (not playing) ---
                    barge_in_frames = 0  # Reset when not playing
                    await stt_service.send_audio(pcm)

                    if speech_prob > SPEECH_ONSET_THRESHOLD:
                        if not is_user_speaking:
                            logger.info(f"User started speaking (prob: {speech_prob:.2f})")
                            is_user_speaking = True

                            # Cancel any pending turn task (shouldn't be playing, but safety check)
                            if current_turn_task and not current_turn_task.done():
                                logger.info("Cancelling active LLM/TTS task")
                                current_turn_task.cancel()
                                try:
                                    await asyncio.wait_for(current_turn_task, timeout=0.1)
                                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                                    pass

                        silence_frames = 0

                    elif is_user_speaking:
                        if speech_prob < SPEECH_OFFSET_THRESHOLD:
                            silence_frames += 1

                        if silence_frames >= SILENCE_THRESHOLD:
                            logger.info("Silence detected - End of turn")
                            is_user_speaking = False
                            silence_frames = 0

                            rtt_start = time.perf_counter()
                            stt_start = time.perf_counter()
                            user_text = await stt_service.get_and_clear_transcript()
                            stt_latency = (time.perf_counter() - stt_start) * 1000

                            # Filter Hallucinations
                            is_hallucination = False
                            if not user_text or len(user_text) < 2:
                                is_hallucination = True
                            else:
                                lower_text = user_text.lower()
                                for phrase in HALLUCINATIONS:
                                    if phrase in lower_text:
                                        logger.info(f"Ignored hallucination: '{user_text}'")
                                        is_hallucination = True
                                        break

                            if user_text and not is_hallucination:
                                logger.info(f"User Said: '{user_text}'")
                                # Launch streaming turn processing in background
                                current_turn_task = asyncio.create_task(process_turn(user_text, stt_latency, rtt_start))
                            else:
                                logger.info("No valid transcript received")

                elif event == "stop":
                    logger.info("Stream stopped")
                    break

        except WebSocketDisconnect:
            pass  # Normal close
        except Exception as e:
            logger.error(f"Error in handle_stream: {e}", exc_info=True)

        finally:
            logger.info("Call ended")
            if current_turn_task and not current_turn_task.done():
                current_turn_task.cancel()
                try:
                    await current_turn_task
                except (asyncio.CancelledError, Exception):
                    pass
            await stt_service.close()
            llm_service.reset_conversation()
            audio_player.stop()
            self.silero_vad.reset_states()
