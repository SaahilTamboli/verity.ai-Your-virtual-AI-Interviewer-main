import asyncio
import time
import sys
import os
import signal
from typing import AsyncGenerator

import numpy as np
import simpleaudio as sa
import pyaudio

from pyht import Client
from pyht.client import TTSOptions
from pyht.protos import api_pb2

from groq import AsyncGroq
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
from deepgramstt import DeepgramSTT

# Load environment variables
load_dotenv()

# Global variables
stream = None
p = None

def check_env_variables():
    required_vars = ["GROQ_API_KEY", "DEEPGRAM_API_KEY", "PLAY_HT_USER_ID", "PLAY_HT_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("Error: The following required environment variables are missing:")
        for var in missing_vars:
            print(f"- {var}")
        print("Please set these variables in your .env file or environment.")
        sys.exit(1)

class DeepgramTranscriber:
    def __init__(self):
        self.dg_stt = DeepgramSTT()
        self.transcript = ""
        self.transcript_ready = asyncio.Event()

    async def process_audio(self):
        if not self.dg_stt.start_connection():
            print("Failed to start Deepgram connection")
            return None
        return self.dg_stt

    def handle_transcript(self):
        if self.dg_stt.transcript_ready.is_set():
            self.transcript = self.dg_stt.final_transcription
            self.transcript_ready.set()
            self.dg_stt.transcript_ready.clear()

async def get_speech_input(connection):
    print("Listening... (Speak your question)")
    transcriber = DeepgramTranscriber()
    dg_connection = await transcriber.process_audio()
    
    if dg_connection is None:
        return None

    timeout = 15  # Increase timeout to 15 seconds
    start_time = time.time()

    try:
        while time.time() - start_time < timeout:
            data = stream.read(3200, exception_on_overflow=False)
            dg_connection.send_audio_data(data)
            
            transcriber.handle_transcript()
            if transcriber.transcript_ready.is_set():
                print(f"You said: {transcriber.transcript}")
                return transcriber.transcript

            await asyncio.sleep(0.05)  # Reduce sleep time for more frequent checks

        print("No speech detected. Please try again.")
        return None
    except Exception as e:
        print(f"Error in speech input: {e}")
        return None

async def get_groq_response_stream(prompt) -> AsyncGenerator[str, None]:
    client = AsyncGroq(api_key="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")
    start_time = time.time()
    
    async for chunk in await client.chat.completions.create(
        messages=[
            {
                "role": "assistant",
                "content": "You are a helpful assistant. Provide concise responses without any formatting. Answer in paragraphs, and in a conversational manner. Do not include any numbers or abbreviations. Use simple sentences."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-70b-versatile",
        stream=True,
    ):
        yield chunk.choices[0].delta.content or ""
    
    end_time = time.time()
    print(f"Groq Latency: {end_time - start_time:.2f} seconds")

async def tts_stream(client, text, options):
    for chunk in client.tts(text, options):
        yield chunk

async def play_audio_stream(stream):
    buff_size = 10485760  # Smaller buffer for faster playback start
    buffer = np.empty(buff_size, np.float16)
    ptr = 0
    audio = None

    async for chunk in stream:
        chunk_samples = np.frombuffer(chunk, np.float16)
        chunk_size = len(chunk_samples)

        if ptr + chunk_size > buff_size:
            if audio is None:
                audio = sa.play_buffer(buffer[:ptr], 1, 2, 24000)
            else:
                audio.wait_done()
                audio = sa.play_buffer(buffer[:ptr], 1, 2, 24000)
            ptr = 0

        buffer[ptr:ptr+chunk_size] = chunk_samples
        ptr += chunk_size

    if ptr > 0:
        if audio is None:
            audio = sa.play_buffer(buffer[:ptr], 1, 2, 24000)
        else:
            audio.wait_done()
            audio = sa.play_buffer(buffer[:ptr], 1, 2, 24000)

    if audio is not None:
        audio.wait_done()

async def process_input(client, options, prompt):
    text_response = ""
    tts_queue = asyncio.Queue()
    
    async def collect_text():
        nonlocal text_response
        async for text_chunk in get_groq_response_stream(prompt):
            text_response += text_chunk
            await tts_queue.put(text_chunk)
        await tts_queue.put(None)  # Signal end of text

    async def generate_and_play_audio():
        buffer = ""
        while True:
            chunk = await tts_queue.get()
            if chunk is None:
                break
            buffer += chunk
            if len(buffer) >= 150 or (chunk is None and buffer):  # Increased buffer size
                audio_stream = tts_stream(client, buffer, options)
                await play_audio_stream(audio_stream)
                buffer = ""
        
        if buffer:  # Play any remaining text
            audio_stream = tts_stream(client, buffer, options)
            await play_audio_stream(audio_stream)

    await asyncio.gather(collect_text(), generate_and_play_audio())
    print("Groq Response:", text_response)

async def initialize_apis(client, options):
    print("Initializing APIs...")
    start_time = time.time()

    # Initialize Groq
    dummy_prompt = "Hello"
    async for _ in get_groq_response_stream(dummy_prompt):
        pass

    # Initialize PlayHT
    dummy_text = "Initialization complete"
    async for _ in tts_stream(client, dummy_text, options):
        pass

    print(f"APIs initialized in {time.time() - start_time:.2f} seconds")

async def main_async():
    global stream, p

    check_env_variables()

    user = "SL3pL2pJ91X5qaunEKsrv4u0Uv53"
    key = "16e69ce24b2f4e16a5440b3603d8e957"
    voice = "s3://voice-cloning-zero-shot/e5df2eb3-5153-40fa-9f6e-6e27bbb7a38e/original/manifest.json"
    quality = "faster"
    use_speech_input = True

    # Setup the client
    try:
        client = Client(user, key)
    except AssertionError as e:
        print(f"Error initializing PlayHT client: {e}")
        print("Please check your PLAYHT_USER_ID and PLAYHT_API_KEY environment variables.")
        return

    # Set the speech options
    options = TTSOptions(voice=voice, format=api_pb2.FORMAT_WAV, quality=quality)

    # Initialize APIs
    await initialize_apis(client, options)

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=3200)

    # Initialize Deepgram
    transcriber = DeepgramTranscriber()
    connection = await transcriber.process_audio()

    if connection is None:
        print("Failed to initialize Deepgram. Exiting.")
        return

    print("Starting interactive session.")
    print("Speak your question or say 'quit' to exit.")
    
    while True:
        try:
            if use_speech_input:
                t = await get_speech_input(connection)
                if t is None:
                    continue
            else:
                t = input("> ")
            if t and t.lower() == 'quit':
                break
            start_time = time.time()
            await process_input(client, options, t)
            print(f"Total response time: {time.time() - start_time:.2f} seconds")
            
            # Reset Deepgram connection after each interaction
            connection.finish()
            connection = await transcriber.process_audio()
            if connection is None:
                print("Failed to reinitialize Deepgram. Exiting.")
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Continuing to next iteration...")
    
    print("Interactive session closed.")

    # Cleanup
    if connection:
        connection.finish()
    stream.stop_stream()
    stream.close()
    p.terminate()
    client.close()

def signal_handler(signum, frame):
    print("Interrupt received, cleaning up...")
    if stream:
        stream.stop_stream()
        stream.close()
    if p:
        p.terminate()
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    asyncio.run(main_async())
    return 0

if __name__ == "__main__":
    sys.exit(main())
