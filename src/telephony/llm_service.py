"""
LLM Service using Groq API.
Provides fast inference for voice agent responses.
"""
import os
from groq import Groq
from src.config import config


class LLMService:
    """Groq LLM service for voice agent conversation."""
    
    def __init__(self):
        """Initialize Groq client and conversation state."""
        api_key = config.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable required")
        
        print(f"🔧 LLMService: Initializing Groq with model: {config.GROQ_LLM_MODEL}")
        self.client = Groq(api_key=api_key)
        self.model = config.GROQ_LLM_MODEL
        
        self.system_prompt = (
            "You are a helpful, concise voice assistant. "
            "Your responses will be spoken aloud, so keep them brief (1-2 sentences) "
            "and avoid using markdown or special characters that are hard to pronounce."
        )
        
        self.greeting = "Hello! I am your AI assistant. How can I help you today?"
        self.conversation_history = []
        self.reset_conversation()

    def get_greeting(self) -> str:
        """Return the initial greeting."""
        return self.greeting

    def generate_response(self, text: str) -> str:
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
            chat_completion = self.client.chat.completions.create(
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
