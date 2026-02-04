from .asr_interface import ASRInterface

class ElevenLabsASR(ASRInterface):
    """
    ASR implementation for ElevenLabs.
    Note: Realtime Scribe v2 is handled via ScribeService in the telephony handler.
    This class exists to satisfy the ASRFactory interface for the main entry point.
    """
    def __init__(self, **kwargs):
        pass

    async def transcribe(self, client):
        """
        Not used in Scribe Realtime flow.
        """
        return {"text": "", "words": []}
