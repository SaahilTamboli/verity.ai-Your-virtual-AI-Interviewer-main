import os
import json
import asyncio
import websockets
import numpy as np
import sounddevice as sd
from typing import AsyncGenerator, AsyncIterable

class AudioManager:
    def __init__(self):
        self.input_stream = None
        self.recording_task = None
        self.stop_recording_event = asyncio.Event()
        self.full_transcript = ""

    def create_input_stream(self):
        if self.input_stream is None:
            self.input_stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16')
            self.input_stream.start()
        return self.input_stream

    def close_streams(self):
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None

    def terminate(self):
        self.close_streams()

    async def get_continuous_speech_input(self):
        url = "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1"
        
        stream = self.create_input_stream()

        print("Listening... (Speak your answer)")

        async with websockets.connect(
            url,
            extra_headers={
                "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}"
            },
        ) as ws:
            self.stop_recording_event.clear()
            self.full_transcript = ""

            async def sender(ws):
                try:
                    while not self.stop_recording_event.is_set():
                        try:
                            data, _ = stream.read(4096)
                            await ws.send(data.tobytes())
                        except OSError as e:
                            print(f"Error reading from audio stream: {e}")
                            break
                        await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    pass
                finally:
                    await ws.send(json.dumps({"type": "CloseStream"}))

            async def receiver(ws):
                async for msg in ws:
                    if self.stop_recording_event.is_set():
                        break
                    res = json.loads(msg)
                    if res.get("is_final"):
                        transcript = res.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                        if transcript:
                            self.full_transcript += " " + transcript
                            print(f"Partial transcript: {transcript}")

            sender_task = asyncio.create_task(sender(ws))
            receiver_task = asyncio.create_task(receiver(ws))

            await asyncio.gather(sender_task, receiver_task, return_exceptions=True)

        return self.full_transcript.strip()

    def start_recording(self):
        self.stop_recording_event.clear()
        self.full_transcript = ""
        self.recording_task = asyncio.create_task(self.get_continuous_speech_input())

    async def stop_recording(self):
        if self.recording_task:
            self.stop_recording_event.set()
            await self.recording_task
            self.recording_task = None
        return self.full_transcript.strip()

audio_manager = AudioManager()

async def async_get_speech_input():
    return await audio_manager.get_continuous_speech_input()

async def async_play_audio(data: AsyncGenerator[bytes, None] | AsyncIterable[bytes]):
    async for chunk in data:
        yield chunk
    print("Audio streaming completed")