"""
FastAPI application for Vobiz telephony integration.
Provides REST and WebSocket endpoints for handling phone calls.
"""
from fastapi import FastAPI, WebSocket
from fastapi.responses import Response

from src.telephony.vobiz_handler import VobizStreamHandler


def create_app(vad_pipeline, asr_pipeline, base_url: str) -> FastAPI:
    """
    Create FastAPI application with Vobiz telephony endpoints.

    Args:
        vad_pipeline: Voice Activity Detection pipeline
        asr_pipeline: Automatic Speech Recognition pipeline
        base_url: Base URL for WebSocket connections (e.g., ws://localhost:8080)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(title="VoiceStreamAI Telephony", version="1.0.0")

    # Store pipelines in app state
    app.state.vad_pipeline = vad_pipeline
    app.state.asr_pipeline = asr_pipeline
    app.state.base_url = base_url

    # Create stream handler
    handler = VobizStreamHandler(vad_pipeline, asr_pipeline)

    @app.get("/")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "service": "voicestreamai-telephony"}

    @app.post("/api/telephony/answer")
    async def answer_call():
        """
        Vobiz webhook endpoint for incoming calls.
        Returns XML to connect the call to our WebSocket.
        """
        import logging
        logger = logging.getLogger(__name__)

        ws_url = app.state.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/api/telephony/stream"

        # XML escape ampersands in URL
        ws_url_escaped = ws_url.replace('&', '&amp;')

        print(f"DEBUG: Answer endpoint called. WebSocket URL: {ws_url_escaped}")
        logger.info(f"Answer endpoint called - WebSocket URL: {ws_url_escaped}")

        # Vobiz XML format: URL is inner content, not attribute!
        xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" keepCallAlive="true" audioTrack="inbound" contentType="audio/x-mulaw;rate=8000" streamTimeout="7200">{ws_url_escaped}</Stream>
</Response>"""

        print(f"DEBUG: Returning XML: {xml_response}")
        logger.info(f"Returning XML: {xml_response}")
        return Response(content=xml_response, media_type="application/xml")

    @app.websocket("/api/telephony/stream")
    async def websocket_stream(websocket: WebSocket):
        """
        WebSocket endpoint for Vobiz audio streaming.
        Receives µ-law audio and returns transcriptions.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"WebSocket connection attempt from {websocket.client}")
        print(f"DEBUG: WebSocket connection attempt from {websocket.client}")
        try:
            await handler.handle_stream(websocket)
        except Exception as e:
            logger.error(f"WebSocket endpoint error: {e}", exc_info=True)

    @app.post("/api/telephony/hangup")
    async def hangup_call():
        """
        Vobiz hangup endpoint for call completion events.
        Returns empty XML response.
        """
        xml_response = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
</Response>"""
        return Response(content=xml_response, media_type="application/xml")

    return app
