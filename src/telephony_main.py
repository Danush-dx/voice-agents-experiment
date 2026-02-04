"""
Entry point for VoiceStreamAI telephony server.
Runs FastAPI server with Vobiz integration alongside the main WebSocket server.
"""
import argparse
import json
import logging
import uvicorn

from src.vad.vad_factory import VADFactory
from src.asr.asr_factory import ASRFactory
from src.telephony.fastapi_app import create_app


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="VoiceStreamAI Telephony Server")

    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--base-url", type=str, required=True,
                        help="Base URL for WebSocket connections (e.g., ws://localhost:8080)")

    # SSL configuration
    parser.add_argument("--ssl-certfile", type=str, help="Path to SSL certificate file")
    parser.add_argument("--ssl-keyfile", type=str, help="Path to SSL key file")

    # VAD configuration
    parser.add_argument("--vad-type", type=str, default="pyannote",
                        choices=["silero", "pyannote", "webrtc"],
                        help="Type of VAD to use")
    parser.add_argument("--vad-args", type=str, default="{}",
                        help="JSON string of VAD-specific arguments")

    # ASR configuration
    parser.add_argument("--asr-type", type=str, default="faster_whisper",
                        choices=["faster_whisper"],
                        help="Type of ASR to use")
    parser.add_argument("--asr-args", type=str, default="{}",
                        help="JSON string of ASR-specific arguments")

    return parser.parse_args()


def main():
    """Main entry point for the telephony server."""
    args = parse_args()

    # Parse JSON arguments
    try:
        vad_args = json.loads(args.vad_args)
        asr_args = json.loads(args.asr_args)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON arguments: {e}")
        return

    logger.info("Initializing VoiceStreamAI Telephony Server...")
    logger.info(f"VAD: {args.vad_type}, ASR: {args.asr_type}")

    # Initialize VAD pipeline
    try:
        vad_pipeline = VADFactory.create_vad_pipeline(args.vad_type, **vad_args)
        logger.info(f"VAD pipeline initialized: {args.vad_type}")
    except Exception as e:
        logger.error(f"Failed to initialize VAD pipeline: {e}")
        return

    # Initialize ASR pipeline
    try:
        asr_pipeline = ASRFactory.create_asr_pipeline(args.asr_type, **asr_args)
        logger.info(f"ASR pipeline initialized: {args.asr_type}")
    except Exception as e:
        logger.error(f"Failed to initialize ASR pipeline: {e}")
        return

    # Create FastAPI app
    app = create_app(vad_pipeline, asr_pipeline, args.base_url)
    logger.info(f"FastAPI app created with base URL: {args.base_url}")

    # Configure uvicorn
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "log_level": "info"
    }

    # Add SSL if configured
    if args.ssl_certfile and args.ssl_keyfile:
        uvicorn_config["ssl_certfile"] = args.ssl_certfile
        uvicorn_config["ssl_keyfile"] = args.ssl_keyfile
        logger.info(f"SSL enabled with cert: {args.ssl_certfile}")

    # Start server
    logger.info(f"Starting telephony server on {args.host}:{args.port}")
    logger.info(f"Vobiz webhook URL: {args.base_url}/api/telephony/answer")

    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
