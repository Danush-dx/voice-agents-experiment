"""
FastAPI application for Vobiz telephony integration.
Provides REST and WebSocket endpoints for handling phone calls.
"""
import logging

from fastapi import FastAPI, WebSocket
from fastapi.responses import Response

from src.telephony.vobiz_handler import VobizStreamHandler

logger = logging.getLogger(__name__)


def create_app(base_url: str) -> FastAPI:
    """
    Create FastAPI application with Vobiz telephony endpoints.

    Args:
        base_url: Base URL for WebSocket connections (e.g., ws://localhost:8080)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(title="VoiceStreamAI Telephony", version="1.0.0")

    # Store base_url in app state
    app.state.base_url = base_url

    # Create stream handler
    handler = VobizStreamHandler()

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
        ws_url = app.state.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/api/telephony/stream"

        # XML escape ampersands in URL
        ws_url_escaped = ws_url.replace('&', '&amp;')

        logger.info(f"Answer endpoint called - WebSocket URL: {ws_url_escaped}")

        # Vobiz XML format: URL is inner content, not attribute!
        xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" keepCallAlive="true" audioTrack="inbound" contentType="audio/x-mulaw;rate=8000" streamTimeout="7200">{ws_url_escaped}</Stream>
</Response>"""

        return Response(content=xml_response, media_type="application/xml")

    @app.websocket("/api/telephony/stream")
    async def websocket_stream(websocket: WebSocket):
        """
        WebSocket endpoint for Vobiz audio streaming.
        Receives µ-law audio and returns transcriptions.
        """
        logger.info(f"WebSocket connection attempt from {websocket.client}")
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
