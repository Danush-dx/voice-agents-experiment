from .faster_whisper_asr import FasterWhisperASR
from .whisper_asr import WhisperASR
from .elevenlabs_asr import ElevenLabsASR
from .groq_asr import GroqASR


class ASRFactory:
    @staticmethod
    def create_asr_pipeline(asr_type, **kwargs):
        if asr_type == "whisper":
            return WhisperASR(**kwargs)
        if asr_type == "faster_whisper":
            return FasterWhisperASR(**kwargs)
        if asr_type == "elevenlabs":
            return ElevenLabsASR(**kwargs)
        if asr_type == "groq":
            return GroqASR(**kwargs)
        else:
            raise ValueError(f"Unknown ASR pipeline type: {asr_type}")
