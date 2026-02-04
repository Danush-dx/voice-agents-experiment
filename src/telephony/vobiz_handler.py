"""
Vobiz WebSocket stream handler.
Handles Vobiz protocol messages and feeds audio to the VAD/ASR pipeline.
"""
import json
import logging
import uuid
from fastapi import WebSocket

from src.client import Client
from src.telephony.audio_converter import AudioConverter


logger = logging.getLogger(__name__)


class VobizStreamHandler:
    """Handles Vobiz WebSocket streams and processes audio through the VAD/ASR pipeline."""

    def __init__(self, vad_pipeline, asr_pipeline):
        """
        Initialize the Vobiz stream handler.

        Args:
            vad_pipeline: Voice Activity Detection pipeline
            asr_pipeline: Automatic Speech Recognition pipeline
        """
        self.vad_pipeline = vad_pipeline
        self.asr_pipeline = asr_pipeline

    async def handle_stream(self, websocket: WebSocket):
        """
        Handle a Vobiz WebSocket stream connection.

        Args:
            websocket: FastAPI WebSocket connection
        """
        await websocket.accept()

        # Create unique client for this call
        client_id = str(uuid.uuid4())
        client = Client(
            client_id=client_id,
            sampling_rate=16000,  # Fixed after conversion
            samples_width=2       # Fixed 16-bit
        )

        # Create audio converter for this call
        converter = AudioConverter()

        stream_id = None
        logger.info(f"Vobiz stream connected: client_id={client_id}")

        try:
            while True:
                # Receive JSON message from Vobiz
                message_text = await websocket.receive_text()

                try:
                    message = json.loads(message_text)
                    event = message.get("event")

                    if event == "connected":
                        logger.info(f"Vobiz connected event received: client_id={client_id}")
                        continue

                    elif event == "start":
                        stream_id = message.get("streamId")
                        logger.info(f"Vobiz stream started: stream_id={stream_id}, client_id={client_id}")
                        continue

                    elif event == "media":
                        # Extract audio payload
                        media = message.get("media", {})
                        payload = media.get("payload")

                        if not payload:
                            logger.warning(f"Empty payload in media event: client_id={client_id}")
                            continue

                        # Convert audio from µ-law 8kHz to PCM 16kHz
                        try:
                            pcm_16khz = converter.convert(payload)

                            # Feed to client and process
                            client.append_audio_data(pcm_16khz)
                            await client.process_audio(websocket, self.vad_pipeline, self.asr_pipeline)

                        except Exception as e:
                            logger.error(f"Audio conversion error: {e}, client_id={client_id}")
                            continue

                    else:
                        logger.warning(f"Unknown event type: {event}, client_id={client_id}")
                        continue

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}, client_id={client_id}")
                    continue

        except Exception as e:
            logger.error(f"WebSocket error: {e}, client_id={client_id}")

        finally:
            logger.info(f"Vobiz stream disconnected: client_id={client_id}, stream_id={stream_id}")
            converter.reset()
