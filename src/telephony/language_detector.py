"""
Language Detection Service using Groq's Llama-3.1-8b-instant.
Optimized for ultra-low latency language classification.
"""
import asyncio
from groq import AsyncGroq
from src.config import config

class LanguageDetector:
    """
    Ultra-fast language detector for text inputs.
    Classifies text into: English (en), Hindi (hi), Tamil (ta), Telugu (te), Kannada (kn).
    """

    def __init__(self):
        """Initialize Groq client for detection."""
        api_key = config.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable required")
        
        self.client = AsyncGroq(api_key=api_key)
        self.model = "llama-3.1-8b-instant" # Fastest model available
        
        # Optimized system prompt for deterministic output
        self.system_prompt = (
            "You are a language classifier. "
            "Classify the input text into one of these languages: "
            "English (en), Hindi (hi), Tamil (ta), Telugu (te), Kannada (kn). "
            "Return ONLY the 2-letter ISO code. "
            "If mixed or unclear, return 'en'. "
            "Examples:\\n"
            "User: 'How are you?' -> en\\n"
            "User: 'Kya haal hai' -> hi\\n"
            "User: 'Tamil theriyuma' -> ta\\n"
            "User: 'Bagunnara' -> te\\n"
            "User: 'Chennagidira' -> kn"
        )

    async def detect_language(self, text: str) -> str:
        """
        Detect language of the text.
        
        Args:
            text: Input text to classify.
            
        Returns:
            2-letter ISO language code (en, hi, ta, te, kn).
        """
        if not text or len(text.strip()) < 2:
            return "en"

        try:
            # aggressive optimization: max_tokens=5, temperature=0.0
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text}
                    ],
                    model=self.model,
                    max_tokens=5,
                    temperature=0.0, 
                    stream=False
                ),
                timeout=0.5 # Strict 500ms timeout
            )
            
            lang_code = response.choices[0].message.content.strip().lower()
            
            # Clean up potential extra chars (like punctuation)
            lang_code = ''.join(filter(str.isalpha, lang_code))
            
            valid_codes = {"en", "hi", "ta", "te", "kn"}
            if lang_code in valid_codes:
                return lang_code
            
            print(f"⚠️ LanguageDetector: Invalid code '{lang_code}', defaulting to 'en'")
            return "en"

        except asyncio.TimeoutError:
            print(f"⚠️ LanguageDetector: Timeout on text '{text[:20]}...', defaulting to 'en'")
            return "en"
        except Exception as e:
            print(f"❌ LanguageDetector error: {e}")
            return "en"

    async def close(self):
        """Close the client."""
        await self.client.close()
