"""
LLM Service using Groq API.
Provides fast inference for voice agent responses.
"""
import os
import asyncio
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
        current_date = "Monday, Dec 23, 2024"
        current_day_of_week = "Monday"
        current_time = "10:00 AM"

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
            f"Available Slots: {available_slots_formatted}.\n"
            f"Current Date: {current_date}.\n"
            f"Current Day of Week: {current_day_of_week}.\n"
            f"Current Time: {current_time}.\n"
            "INSTRUCTIONS:\n"
            "1. Speak naturally with a South Indian English accent (use 'only', 'itself', 'vill' for will).\n"
            "2. Keep responses SHORT (1-2 sentences max).\n"
            "3. LANGUAGE SWITCHING: DEFAULT TO ENGLISH. Only switch to Hindi/Tamil/Kannada/Telugu if the user speaks a COMPLETE phrase in that language. If unsure, speak English.\n"
            "4. NEVER mix languages. Do not say 'Hindi. Aap kaise hain'. Just say 'Aap kaise hain'.\n"
            "5. CALL FLOW: Ask for Day -> Ask for Time -> Share Price -> Get Buyer Offer.\n"
            "6. IGNORE background noise or short mumbles.\n"
            "7. If asked about EMI, say finance team will help at the center."
        )

        self.greeting = f"Hello {customer_name}... This is Mira speaking from DriveX... I am calling regarding your interest in the {vehicle_model} vehicle... The seller has invited you for a free test drive... Can I book the appointment for you sir?"
        self.conversation_history = []
        self.reset_conversation()

    def get_greeting(self) -> str:
        """Return the initial greeting."""
        return self.greeting

    async def generate_response(self, text: str) -> str:
        """
        Generate LLM response using Groq (Non-streaming).
        """
        if not text or not text.strip():
            return ""

        print(f"🤖 LLMService: Generating response for: '{text}'")
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": text})
        
        try:
            # Add timeout to prevent hangs
            chat_completion = await asyncio.wait_for(
                self.client.chat.completions.create(
                    messages=self.conversation_history,
                    model=self.model,
                    max_tokens=200,
                    temperature=0.4,
                ),
                timeout=15.0 # 15s timeout
            )
            
            response = chat_completion.choices[0].message.content.strip()
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except asyncio.TimeoutError:
            print("❌ LLMService error: Request timed out")
            return "Could you please say that again?"
        except Exception as e:
            print(f"❌ LLMService error: {e}")
            import traceback
            traceback.print_exc()
            return "I'm sorry, I didn't quite catch that."

    async def generate_stream(self, text: str):
        """
        Generate LLM response as a stream of chunks.
        Yields text fragments as they arrive.
        """
        if not text or not text.strip():
            return

        print(f"🤖 LLMService: Streaming response for: '{text}'")
        self.conversation_history.append({"role": "user", "content": text})
        
        full_response = ""
        try:
            stream = await self.client.chat.completions.create(
                messages=self.conversation_history,
                model=self.model,
                max_tokens=200,
                temperature=0.4,
                stream=True,
            )
            
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content
            
            # Save full response to history
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            print(f"❌ LLMService stream error: {e}")
            yield "I'm sorry, I'm having trouble connecting."

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        print("🔄 LLMService: Conversation history reset")
