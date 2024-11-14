import base64
import textwrap
from pyht import TTSOptions
from pyht.async_client import AsyncClient
from pyht.protos import api_pb2
import os
import logging
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        logger.info("Initializing AudioProcessor")
        self.client = AsyncClient(os.getenv('PLAY_HT_USER_ID'), os.getenv('PLAY_HT_API_KEY'))
        self.options = TTSOptions(
            voice="s3://voice-cloning-zero-shot/e5df2eb3-5153-40fa-9f6e-6e27bbb7a38e/original/manifest.json",
            format=api_pb2.FORMAT_WAV,
            quality="standard",
            speed=0.8,
            sample_rate=48000,
            voice_guidance=0.5,
            temperature=0.5
        )

    async def initialize(self):
        logger.info("AudioProcessor initialized successfully")

    async def process_response(self, response):
        logger.info(f"Processing audio response of length: {len(response)}")
        chunks = textwrap.wrap(response, 500, break_long_words=False, replace_whitespace=False)
        for chunk in chunks:
            logger.debug(f"Processing chunk: {chunk}")
            try:
                async for audio_chunk in self.client.tts(chunk, self.options):
                    # Convert the audio chunk to a Float32Array
                    float_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    yield float_array.tobytes()
            except Exception as e:
                logger.error(f"Error generating audio for chunk: {str(e)}", exc_info=True)
        logger.info("Audio response processing completed")

    async def terminate(self):
        logger.info("AudioProcessor terminated successfully")
