import os
import json
import asyncio
import websockets
import numpy as np
import sounddevice as sd
from typing import AsyncGenerator, AsyncIterable
import logging

logger = logging.getLogger(__name__)

class AudioManager:
    def __init__(self):
        logger.info("Initializing AudioManager")
        self.input_stream = None
        self.recording_task = None
        self.stop_recording_event = asyncio.Event()
        self.full_transcript = ""

    async def initialize(self):
        logger.info("AudioManager initialized successfully")

    def create_input_stream(self):
        if self.input_stream is None:
            logger.info("Creating new input stream")
            self.input_stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16')
            self.input_stream.start()
        return self.input_stream

    def close_streams(self):
        if self.input_stream:
            logger.info("Closing input stream")
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None

    async def terminate(self):
        logger.info("Terminating AudioManager")
        self.close_streams()

    async def get_continuous_speech_input(self):
        logger.info("Starting continuous speech input")
        url = "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1&punctuate=true&language=en-IN"
        
        stream = self.create_input_stream()

        async with websockets.connect(
            url,
            extra_headers={
                "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}",
                "Model": "nova-2",
                "Smart_format": "true",
                "Punctuation": "true",
                "Language": "en-IN",
            },
        ) as ws:
            self.stop_recording_event.clear()
            self.full_transcript = ""

            async def sender(ws):
                try:
                    while not self.stop_recording_event.is_set():
                        data, _ = stream.read(4096)
                        await ws.send(data.tobytes())
                        await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    logger.info("Audio sender cancelled")
                except Exception as e:
                    logger.error(f"Error in audio sender: {str(e)}", exc_info=True)
                finally:
                    await ws.send(json.dumps({"type": "CloseStream"}))

            async def receiver(ws):
                try:
                    async for msg in ws:
                        if self.stop_recording_event.is_set():
                            break
                        res = json.loads(msg)
                        if res.get("is_final"):
                            transcript = res.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                            if transcript:
                                logger.debug(f"Received transcript: {transcript}")
                                self.full_transcript += " " + transcript
                except Exception as e:
                    logger.error(f"Error in audio receiver: {str(e)}", exc_info=True)

            await asyncio.gather(sender(ws), receiver(ws))

        logger.info("Continuous speech input completed")
        return self.full_transcript.strip()

    def start_recording(self):
        logger.info("Starting audio recording")
        self.stop_recording_event.clear()
        self.full_transcript = ""
        self.recording_task = asyncio.create_task(self.get_continuous_speech_input())

    async def stop_recording(self):
        logger.info("Stopping audio recording")
        if self.recording_task:
            self.stop_recording_event.set()
            await self.recording_task
            self.recording_task = None
        logger.info(f"Recording stopped. Transcript length: {len(self.full_transcript)}")
        return self.full_transcript.strip()

audio_manager = AudioManager()

async def async_get_speech_input():
    return await audio_manager.get_continuous_speech_input()

async def async_play_audio(data: AsyncGenerator[bytes, None] | AsyncIterable[bytes]):
    async for chunk in data:
        yield chunk
    logger.info("Audio streaming completed")
