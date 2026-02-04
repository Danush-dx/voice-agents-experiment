"""
Centralized configuration management for VoiceStreamAI.
Loads configuration from environment variables and .env file.
"""
import os
from pathlib import Path
from typing import Optional
import logging

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    logging.info(f"Loaded configuration from {env_path}")
except ImportError:
    logging.warning("python-dotenv not installed. Using system environment variables only.")
    pass


class Config:
    """Configuration class for VoiceStreamAI."""

    # Server Configuration
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', '8080'))
    BASE_URL: Optional[str] = os.getenv('BASE_URL')

    # HuggingFace Configuration
    HF_TOKEN: Optional[str] = os.getenv('HF_TOKEN')

    # VAD Configuration
    VAD_TYPE: str = os.getenv('VAD_TYPE', 'pyannote')

    # ASR Configuration
    ASR_TYPE: str = os.getenv('ASR_TYPE', 'faster_whisper')
    ASR_MODEL_SIZE: str = os.getenv('ASR_MODEL_SIZE', 'large-v3')
    ASR_DEVICE: str = os.getenv('ASR_DEVICE', 'cpu')
    ASR_COMPUTE_TYPE: str = os.getenv('ASR_COMPUTE_TYPE', 'int8')

    # SSL Configuration
    SSL_CERTFILE: Optional[str] = os.getenv('SSL_CERTFILE')
    SSL_KEYFILE: Optional[str] = os.getenv('SSL_KEYFILE')

    # WebSocket Server Configuration (for browser clients)
    WS_HOST: str = os.getenv('WS_HOST', 'localhost')
    WS_PORT: int = int(os.getenv('WS_PORT', '8765'))

    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')

    @classmethod
    def get_vad_args(cls) -> dict:
        """Get VAD-specific arguments based on VAD type."""
        if cls.VAD_TYPE == 'pyannote':
            if not cls.HF_TOKEN:
                raise ValueError("HF_TOKEN is required for PyAnnote VAD. Set it in .env file.")
            return {"auth_token": cls.HF_TOKEN}
        elif cls.VAD_TYPE == 'silero':
            return {}
        elif cls.VAD_TYPE == 'webrtc':
            return {}
        else:
            return {}

    @classmethod
    def get_asr_args(cls) -> dict:
        """Get ASR-specific arguments based on ASR type."""
        if cls.ASR_TYPE == 'faster_whisper':
            return {
                "model_size": cls.ASR_MODEL_SIZE,
                "device": cls.ASR_DEVICE,
                "compute_type": cls.ASR_COMPUTE_TYPE
            }
        else:
            return {}

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        errors = []

        if not cls.BASE_URL:
            errors.append("BASE_URL is required. Set it in .env file or as environment variable.")

        if cls.VAD_TYPE == 'pyannote' and not cls.HF_TOKEN:
            errors.append("HF_TOKEN is required for PyAnnote VAD. Set it in .env file.")

        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    @classmethod
    def display(cls) -> None:
        """Display current configuration (hiding sensitive data)."""
        print("\n" + "="*60)
        print("VoiceStreamAI Configuration")
        print("="*60)
        print(f"Server Host:          {cls.HOST}")
        print(f"Server Port:          {cls.PORT}")
        print(f"Base URL:             {cls.BASE_URL}")
        print(f"VAD Type:             {cls.VAD_TYPE}")
        print(f"ASR Type:             {cls.ASR_TYPE}")
        print(f"ASR Model:            {cls.ASR_MODEL_SIZE}")
        print(f"ASR Device:           {cls.ASR_DEVICE}")
        print(f"ASR Compute Type:     {cls.ASR_COMPUTE_TYPE}")
        print(f"HF Token:             {'***' + cls.HF_TOKEN[-4:] if cls.HF_TOKEN else 'Not Set'}")
        print(f"SSL Enabled:          {'Yes' if cls.SSL_CERTFILE else 'No'}")
        print(f"Log Level:            {cls.LOG_LEVEL}")
        print("="*60)
        print(f"Webhook URL:          {cls.BASE_URL}/api/telephony/answer" if cls.BASE_URL else "Not Set")
        print("="*60 + "\n")


# Create a singleton instance
config = Config()
