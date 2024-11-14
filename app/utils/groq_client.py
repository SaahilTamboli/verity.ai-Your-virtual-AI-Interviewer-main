import os
from groq import Groq
import asyncio
import logging

logger = logging.getLogger(__name__)

class GroqClient:
    def __init__(self):
        logger.info("Initializing GroqClient")
        self.client = None

    async def initialize(self):
        try:
            self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
            logger.info("GroqClient initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GroqClient: {str(e)}", exc_info=True)
            raise

    async def get_response(self, prompt):
        if not self.client:
            logger.warning("GroqClient not initialized. Initializing now.")
            await self.initialize()

        try:
            logger.info(f"Sending prompt to Groq. Prompt length: {len(prompt)}")
            chat_completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.1-70b-versatile",
            )
            response = chat_completion.choices[0].message.content
            logger.info(f"Received response from Groq. Response length: {len(response)}")
            return response
        except Exception as e:
            logger.error(f"Error getting response from Groq: {str(e)}", exc_info=True)
            return "I apologize, but I'm having trouble generating a response at the moment. Let's move on to the next question."

    async def close(self):
        logger.info("Closing GroqClient")
        self.client = None
