"""
Centralized configuration management for VoiceStreamAI.
Loads credentials from environment variables and hardcodes model specifications.
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
    ASR_TYPE: str = os.getenv('ASR_TYPE', 'elevenlabs')
    ASR_MODEL_SIZE: str = os.getenv('ASR_MODEL_SIZE', 'large-v3')
    ASR_DEVICE: str = os.getenv('ASR_DEVICE', 'cpu')
    ASR_COMPUTE_TYPE: str = os.getenv('ASR_COMPUTE_TYPE', 'int8')

    # ElevenLabs Scribe Configuration (Spec: ElevenLabs Scribe v2 realtime)
    ELEVENLABS_API_KEY: Optional[str] = os.getenv('ELEVENLABS_API_KEY')
    ELEVENLABS_STT_MODEL: str = "scribe_v2_realtime"

    # Groq LLM Configuration (Spec: Groq llama-3.1-8b-instant)
    GROQ_API_KEY: Optional[str] = os.getenv('GROQ_API_KEY')
    GROQ_LLM_MODEL: str = "llama-3.1-8b-instant"

    # Cartesia TTS Configuration (Spec: Cartesia Sonic 3)
    CARTESIA_API_KEY: Optional[str] = os.getenv('CARTESIA_API_KEY')
    CARTESIA_TTS_MODEL: str = "sonic-3"
    CARTESIA_VOICE_ID: str = "47f3bbb1-e98f-4e0c-92c5-5f0325e1e206"

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
        return {}

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        errors = []
        if not cls.BASE_URL:
            errors.append("BASE_URL is required.")
        if not cls.ELEVENLABS_API_KEY:
            errors.append("ELEVENLABS_API_KEY is required.")
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required.")
        if not cls.CARTESIA_API_KEY:
            errors.append("CARTESIA_API_KEY is required.")
        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    @classmethod
    def display(cls) -> None:
        """Display current configuration."""
        print("\n" + "="*60)
        print("VoiceStreamAI Configuration (Agent Spec Active)")
        print("="*60)
        print(f"Server Host:          {cls.HOST}")
        print(f"Server Port:          {cls.PORT}")
        print(f"Base URL:             {cls.BASE_URL}")
        print(f"STT Model:            {cls.ELEVENLABS_STT_MODEL}")
        print(f"LLM Model:            {cls.GROQ_LLM_MODEL}")
        print(f"TTS Model:            {cls.CARTESIA_TTS_MODEL}")
        print(f"Log Level:            {cls.LOG_LEVEL}")
        print("="*60 + "\n")


# Create a singleton instance
config = Config()