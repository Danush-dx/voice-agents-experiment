import os
from groq import Groq
from src.audio_utils import save_audio_to_file
from .asr_interface import ASRInterface

class GroqASR(ASRInterface):
    def __init__(self, **kwargs):
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for GroqASR")
        self.client = Groq(api_key=api_key)
        self.model = kwargs.get("model", "whisper-large-v3")

    async def transcribe(self, client):
        file_path = await save_audio_to_file(
            client.scratch_buffer, client.get_file_name()
        )
        
        try:
            with open(file_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(file_path, file.read()),
                    model=self.model,
                    response_format="verbose_json",
                    language=client.config.get("language") # Optional: pass language if known
                )
            
            # Groq's verbose_json response structure:
            # {
            #   "text": "...",
            #   "language": "en",
            #   "duration": ...,
            #   "segments": [ ... ]
            # }
            # Note: Groq might not return word-level timestamps in the same way as Whisper.
            # We'll adapt based on the response.
            
            return {
                "language": transcription.language,
                "language_probability": None, # Groq might not provide this
                "text": transcription.text,
                "words": [] # detailed word timestamps might need extra parsing if available
            }

        except Exception as e:
            print(f"Error in Groq transcription: {e}")
            return {
                "language": "unknown",
                "language_probability": 0.0,
                "text": "",
                "words": []
            }
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
