"""
Entry point for VoiceStreamAI telephony server.
Runs FastAPI server with Vobiz integration alongside the main WebSocket server.
"""
import argparse
import sys
import logging
import uvicorn

from src.config import config
from src.telephony.fastapi_app import create_app


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments. Defaults are loaded from config/env."""
    parser = argparse.ArgumentParser(
        description="VoiceStreamAI Telephony Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration is loaded from .env file and can be overridden by command-line arguments.
Create a .env file from .env.example and set your credentials there.
        """
    )

    # Server configuration
    parser.add_argument("--host", type=str, default=config.HOST,
                        help=f"Host to bind to (default: {config.HOST})")
    parser.add_argument("--port", type=int, default=config.PORT,
                        help=f"Port to bind to (default: {config.PORT})")
    parser.add_argument("--base-url", type=str, default=config.BASE_URL,
                        help="Base URL for WebSocket connections (default: from .env)")

    # SSL configuration
    parser.add_argument("--ssl-certfile", type=str, default=config.SSL_CERTFILE,
                        help="Path to SSL certificate file")
    parser.add_argument("--ssl-keyfile", type=str, default=config.SSL_KEYFILE,
                        help="Path to SSL key file")

    return parser.parse_args()


def main():
    """Main entry point for the telephony server."""
    args = parse_args()

    # Validate configuration
    if not args.base_url:
        logger.error("BASE_URL is required. Set it in .env file or use --base-url flag")
        sys.exit(1)

    # Display configuration
    config.display()

    logger.info("Initializing VoiceStreamAI Telephony Server...")

    # Create FastAPI app
    app = create_app(args.base_url)
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
