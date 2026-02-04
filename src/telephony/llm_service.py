"""
LLM Service using Groq API.
Provides fast inference for voice agent responses.
"""
import os
from groq import AsyncGroq
from src.config import config


class LLMService:
    """Groq LLM service for voice agent conversation."""
    
    def __init__(self):
        """Initialize Groq client and conversation state."""
        api_key = config.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable required")
        
        print(f"🔧 LLMService: Initializing Groq with model: {config.GROQ_LLM_MODEL}")
        self.client = AsyncGroq(api_key=api_key)
        self.model = config.GROQ_LLM_MODEL
        
        self.system_prompt = (
            "You are Mira from DriveX, calling about a Royal Enfield Classic 350. "
            "Keep responses to 1-2 sentences and avoid special characters. "
            "The test drive is at our Koramangala office, located at DriveX Niax Motors, "
            "No. 1004, 80 Feet Road, Koramangala 1st Block, near the Wipro Signal. "
            "The seller expectation is one lakh forty-five thousand rupees. "
            "Ask the customer which day and time works best for their test drive. "
            "Ignore background noise like 'Hello' repetitions. "
            "DETECT the language (English, Hindi, Tamil, Kannada, or Telugu) from the "
            "user's first reply and LOCK into that language for the entire call. "
            "Do not mix languages. If they ask about EMI, say our finance team will help "
            "during the visit. Do not speak anything in brackets."
        )
        
        customer_name = "Danush"
        vehicle_model = "Royal Enfield Classic 350"
        self.greeting = f"Hello {customer_name}... This is Mira speaking from DriveX... I am calling regarding your interest in the {vehicle_model} vehicle... The seller has invited you for a free test drive... Can I book the appointment for you sir?"
        self.conversation_history = []
        self.reset_conversation()

    def get_greeting(self) -> str:
        """Return the initial greeting."""
        return self.greeting

    async def generate_response(self, text: str) -> str:
        """
        Generate LLM response using Groq.
        
        Args:
            text: User transcription
            
        Returns:
            Assistant response string
        """
        if not text or not text.strip():
            return ""

        print(f"🤖 LLMService: Generating response for: '{text}'")
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": text})
        
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=self.conversation_history,
                model=self.model,
                max_tokens=150,
                temperature=0.7,
            )
            
            response = chat_completion.choices[0].message.content.strip()
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            print(f"❌ LLMService error: {e}")
            return "I'm sorry, I'm having trouble thinking right now."

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        print("🔄 LLMService: Conversation history reset")
