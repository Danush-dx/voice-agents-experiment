"""
LLM Service using Groq API.
Provides fast inference for voice agent responses.
"""
import asyncio
import logging
from datetime import datetime
from groq import AsyncGroq
from src.config import config

logger = logging.getLogger(__name__)

MAX_HISTORY = 20


class LLMService:
    """Groq LLM service for voice agent conversation."""

    def __init__(self):
        """Initialize Groq client and conversation state."""
        api_key = config.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable required")

        logger.info(f"LLMService: Initializing Groq with model: {config.GROQ_LLM_MODEL}")
        self.client = AsyncGroq(api_key=api_key)
        self.model = config.GROQ_LLM_MODEL

        # Mock Data for Context
        customer_name = "Ravi"
        customer_phone = "+91-9876543210"
        customer_city = "Bangalore"
        customer_state = "Karnataka"
        vehicle_model = "Royal Enfield Classic 350"
        vehicle_id = "KA-05-AB-1234"
        seller_name = "Danush"
        seller_id = "SLR-101"
        seller_expectation = "one lakh forty-five thousand rupees"
        vehicle_address = "DriveX Niax Motors, No. 1004, 80 Feet Road, Koramangala 1st Block, near Wipro Signal"
        available_slots_formatted = (
            "- Monday (Feb 23): 10 AM - 12 PM, 2 PM - 4 PM\n"
            "- Tuesday (Feb 24): 9 AM - 11 AM\n"
            "- Wednesday (Feb 25): 10 AM - 1 PM"
        )

        # Dynamic date/time
        now = datetime.now()
        current_date = now.strftime("%A, %b %d, %Y")
        current_day_of_week = now.strftime("%A")
        current_time = now.strftime("%I:%M %p")

        self.system_prompt = (
            f"Role: Mira, DriveX Appointment Specialist (South Indian Female Voice Persona).\n"
            f"Goal: Book a test drive for {vehicle_model} at DriveX Koramangala.\n"
            f"Address: {vehicle_address}.\n"
            f"Seller Price: {seller_expectation}.\n"
            f"Available Slots: {available_slots_formatted}.\n"
            f"Current Date: {current_date}.\n"
            f"Current Day of Week: {current_day_of_week}.\n"
            f"Current Time: {current_time}.\n"
            f"Customer Name: {customer_name}.\n"
            f"Customer Phone: {customer_phone}.\n"
            f"Customer City: {customer_city}.\n"
            f"Customer State: {customer_state}.\n"
            f"Customer Vehicle ID: {vehicle_id}.\n"
            f"Seller Name: {seller_name}.\n"
            f"Seller ID: {seller_id}.\n"
            f"Seller Expectation: {seller_expectation}.\n"
            f"Vehicle Address: {vehicle_address}.\n"
            "INSTRUCTIONS:\n"
            "1. You are Mira, a friendly and human-like agent. Treat the user as a friend, not a checklist.\n"
            "2. You have ALREADY introduced yourself in the greeting. DO NOT say 'Hello', 'Hi', or your name again at the start of your response. Just answer the user directly.\n"
            "3. Speak naturally with a gentle South Indian English accent (use 'only', 'itself' occasionally, but keep it subtle).\n"
            "4. Keep responses SHORT (1-2 sentences max). Be concise.\n"
            "5. If the user asks 'When can I come?', immediately offer the specific available slots (e.g., 'We have slots on Monday at 10 AM or Tuesday at 9 AM. Which works for you?'). Do not ask vague questions like 'Which day you prefer?'.\n"
            "6. LANGUAGE SWITCHING & SUPPORTED LANGUAGES:\n"
            "   - You MUST support: English, Hindi, Tamil, Kannada, and Telugu.\n"
            "   - DETECT the user's language based on their input text.\n"
            "   - IF user speaks Hindi -> Reply in HINDI.\n"
            "   - IF user speaks Tamil -> Reply in TAMIL.\n"
            "   - IF user speaks Kannada -> Reply in KANNADA.\n"
            "   - IF user speaks Telugu -> Reply in TELUGU.\n"
            "   - IF user speaks English -> Reply in ENGLISH.\n"
            "   - SWITCH IMMEDIATELY. Do not ask 'Should I speak in Tamil?'. Just do it.\n"
            "7. IGNORE background noise.\n"
        )

        self.greeting = f"Hello {customer_name}... This is Mira speaking from DriveX... I am calling regarding your interest in the {vehicle_model} vehicle... The seller has invited you for a free test drive... Can I book the appointment for you sir?"
        self.conversation_history = []
        self.reset_conversation()

    def get_greeting(self) -> str:
        """Return the initial greeting."""
        return self.greeting

    async def generate_response(self, text: str, language: str = "en") -> str:
        """
        Generate LLM response using Groq (Non-streaming).

        Args:
            text: User input text
            language: Target language code (en, hi, ta, te, kn)
        """
        if not text or not text.strip():
            return ""

        logger.debug(f"LLMService: Generating response for: '{text}' in '{language}'")

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": text})

        # Trim history if too long
        if len(self.conversation_history) > MAX_HISTORY:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-(MAX_HISTORY - 1):]

        language_map = {
            "en": "English",
            "hi": "Hindi",
            "ta": "Tamil",
            "te": "Telugu",
            "kn": "Kannada"
        }
        target_lang_name = language_map.get(language, "English")

        # Create a shallow copy of history to inject the language instruction
        messages = list(self.conversation_history)
        messages.append({
            "role": "system",
            "content": f"IMPORTANT: User is speaking {target_lang_name}. REPLY ONLY IN {target_lang_name}."
        })

        try:
            # Add timeout to prevent hangs
            chat_completion = await asyncio.wait_for(
                self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_tokens=200,
                    temperature=0.4,
                ),
                timeout=15.0  # 15s timeout
            )

            response = chat_completion.choices[0].message.content.strip()

            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response})

            return response

        except asyncio.TimeoutError:
            logger.error("LLMService error: Request timed out")
            return "Could you please say that again?"
        except Exception as e:
            logger.error(f"LLMService error: {e}", exc_info=True)
            return "I'm sorry, I didn't quite catch that."

    async def generate_stream(self, text: str, language: str = "en"):
        """
        Generate LLM response as a stream of token chunks.
        Yields text fragments as they arrive.

        On CancelledError (barge-in), saves partial response to history
        for conversational continuity.
        """
        if not text or not text.strip():
            return

        logger.debug(f"LLMService: Streaming response for: '{text}' in '{language}'")
        self.conversation_history.append({"role": "user", "content": text})

        # Trim history: keep system prompt + last (MAX_HISTORY - 1) messages
        if len(self.conversation_history) > MAX_HISTORY:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-(MAX_HISTORY - 1):]

        # Inject language instruction
        language_map = {
            "en": "English",
            "hi": "Hindi",
            "ta": "Tamil",
            "te": "Telugu",
            "kn": "Kannada"
        }
        target_lang_name = language_map.get(language, "English")

        messages = list(self.conversation_history)
        messages.append({
            "role": "system",
            "content": f"IMPORTANT: User is speaking {target_lang_name}. REPLY ONLY IN {target_lang_name}."
        })

        full_response = ""
        try:
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_tokens=200,
                    temperature=0.4,
                    stream=True,
                ),
                timeout=15.0
            )

            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content

            # Save full response to history
            self.conversation_history.append({"role": "assistant", "content": full_response})

        except asyncio.CancelledError:
            # Barge-in: save partial response for conversational continuity
            if full_response:
                self.conversation_history.append({"role": "assistant", "content": full_response + "..."})
                logger.info(f"LLMService: Barge-in, saved partial response ({len(full_response)} chars)")
            raise  # Re-raise so caller knows it was cancelled
        except asyncio.TimeoutError:
            logger.error("LLMService stream error: Request timed out")
            yield "Could you please say that again?"
        except Exception as e:
            logger.error(f"LLMService stream error: {e}", exc_info=True)
            yield "I'm sorry, I'm having trouble connecting."

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "assistant", "content": self.greeting}
        ]
        logger.info("LLMService: Conversation history reset")
