from __future__ import annotations
import select
import sys
from typing import AsyncGenerator, AsyncIterable, Generator, Iterable, Literal

import asyncio
import time
import os
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
import pyaudio

import numpy as np
import sounddevice as sd
from pyht import TTSOptions
import simpleaudio as sa
import websockets
import json

from pyht.async_client import AsyncClient
from pyht.protos import api_pb2

from groq import Groq
from dotenv import load_dotenv

from contextual_memory import create_conversation_chain, get_response
from rag_pipeline import initialize_rag_system, get_relevant_information, generate_interview_response
import threading
import queue

# Load environment variables
load_dotenv()

# === SYNC EXAMPLE ===


def play_audio(data: Generator[bytes, None, None] | Iterable[bytes]):
    buff_size = 5242880
    ptr = 0
    start_time = time.time()
    buffer = np.empty(buff_size, np.float16)
    audio = None
    for i, chunk in enumerate(data):
        if i == 0:
            start_time = time.time()
            continue  # Drop the first response, we don't want a header.
        elif i == 1:
            print("First audio byte received in:", time.time() - start_time)
        for sample in np.frombuffer(chunk, np.float16):
            buffer[ptr] = sample
            ptr += 1
        if i == 5:
            # Give a 4 sample worth of breathing room before starting
            # playback
            audio = sa.play_buffer(buffer, 1, 2, 24000)
    approx_run_time = ptr / 24_000
    time.sleep(max(approx_run_time - time.time() + start_time, 0))
    if audio is not None:
        audio.stop()

def get_groq_response(prompt):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    start_time = time.time()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide concise responses without any formatting."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-70b-versatile",
    )
    end_time = time.time()
    
    response = chat_completion.choices[0].message.content
    latency = end_time - start_time
    
    print(f"Groq Latency: {latency:.2f} seconds")
    return response


async def initialize_apis(client, options):
    print("Initializing APIs...")
    start_time = time.time()

    # Initialize Groq
    dummy_prompt = "Hello"
    await asyncio.to_thread(get_groq_response, dummy_prompt)

    # Initialize PlayHT
    dummy_text = "Initialization complete"
    async for _ in client.tts(dummy_text, options):
        pass  # We just need to iterate through the generator

    print(f"APIs initialized in {time.time() - start_time:.2f} seconds")


class AudioManager:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.input_stream = None

    def create_input_stream(self):
        if self.input_stream is None:
            self.input_stream = self.p.open(format=pyaudio.paInt16,
                                            channels=1,
                                            rate=16000,
                                            input=True,
                                            frames_per_buffer=1024)
        return self.input_stream

    def close_streams(self):
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None

    def terminate(self):
        self.close_streams()
        self.p.terminate()

audio_manager = AudioManager()

async def async_get_speech_input():
    url = "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1"
    
    stream = audio_manager.create_input_stream()

    print("Press Enter to start speaking, and press Enter again to end your answer.")

    def input_thread(input_queue):
        while True:
            input_queue.put(input())

    input_queue = queue.Queue()
    input_thread = threading.Thread(target=input_thread, args=(input_queue,), daemon=True)
    input_thread.start()

    try:
        async with websockets.connect(
            url,
            extra_headers={
                "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}"
            },
        ) as ws:
            async def sender(ws):
                try:
                    input_queue.get()  # Wait for Enter to start recording
                    print("Recording started. Speak your answer...")
                    while True:
                        data = stream.read(1024, exception_on_overflow=False)
                        await ws.send(data)
                        if not input_queue.empty():
                            print("Recording ended.")
                            break
                        await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"Error in sender: {str(e)}")
                finally:
                    await ws.send(json.dumps({"type": "CloseStream"}))

            async def receiver(ws):
                transcript = ""
                try:
                    async for msg in ws:
                        res = json.loads(msg)
                        if res.get("is_final"):
                            transcript = res.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                            print(f"Partial transcript: {transcript}")
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    print(f"Error in receiver: {str(e)}")
                return transcript

            sender_task = asyncio.create_task(sender(ws))
            receiver_task = asyncio.create_task(receiver(ws))
            
            done, pending = await asyncio.wait(
                [sender_task, receiver_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in pending:
                task.cancel()
            
            transcript = receiver_task.result() if receiver_task in done else ""

        print(f"Final transcript: {transcript}")
        return transcript
    except Exception as e:
        print(f"Error in async_get_speech_input: {str(e)}")
        return ""

async def async_play_audio(data: AsyncGenerator[bytes, None] | AsyncIterable[bytes]):
    buffer = np.array([], dtype=np.int16)
    start_time = time.time()
    stream = None
    first_chunk = True
    total_samples = 0

    async for chunk in data:
        if first_chunk:
            print("First audio byte received in:", time.time() - start_time)
            first_chunk = False
        
        chunk_array = np.frombuffer(chunk, dtype=np.int16)
        buffer = np.append(buffer, chunk_array)
        total_samples += len(chunk_array)
        print(f"Received chunk of size: {len(chunk_array)} samples")

        if len(buffer) >= 7200:  # Start playing after accumulating 0.2 seconds of audio
            if stream is None:
                stream = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
                stream.start()
            
            stream.write(buffer)
            buffer = np.array([], dtype=np.int16)

    # Play any remaining audio
    if len(buffer) > 0 and stream is not None:
        stream.write(buffer)

    if stream:
        stream.stop()
        stream.close()

    print(f"Total audio samples: {total_samples}")
    print(f"Audio duration: {total_samples / 24000:.2f} seconds")
    print("Audio playback completed")

async def initialize_apis(client, options):
    print("Initializing APIs...")
    start_time = time.time()

    # Initialize Groq
    dummy_prompt = "Hello"
    await asyncio.to_thread(get_groq_response, dummy_prompt)

    # Initialize PlayHT
    dummy_text = "Initialization complete"
    async for _ in client.tts(dummy_text, options):
        pass  # We just need to iterate through the generator

    print(f"APIs initialized in {time.time() - start_time:.2f} seconds")

async def main():
    # Load values from environment variables
    user = os.getenv("PLAY_HT_USER_ID")
    key = os.getenv = ("PLAY_HT_API_KEY")
    voice = "s3://voice-cloning-zero-shot/e5df2eb3-5153-40fa-9f6e-6e27bbb7a38e/original/manifest.json"
    quality = "standard"
    interactive = True
    use_speech_input = True

    # Setup the client
    client = AsyncClient(user, key)

    # Set the speech options
    options = TTSOptions(voice=voice, format=api_pb2.FORMAT_WAV, quality=quality, speed=0.8, sample_rate=24000, voice_guidance=0.5, temperature=0.5)

    # Initialize APIs
    await initialize_apis(client, options)

    # Initialize the RAG system
    docsearch = initialize_rag_system()

    # Initialize the conversation chain
    conversation = create_conversation_chain(os.getenv('GROQ_API_KEY'))

    # Initialize Groq client
    groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    model = "llama-3.1-70b-versatile"

    # Load the system prompt from an external file
    with open("interviewer_system_prompt.txt", "r") as f:
        system_prompt = f.read()

    print("Starting the interview. The AI interviewer will ask you questions based on your resume.")
    context = "This is an AI-powered interview for a software engineering position. Ask relevant questions based on the candidate's resume and the job description."

    try:
        # Welcome message
        welcome_message = "Welcome to the interview. I'll be asking you a series of questions to assess your suitability for this role. Please press Enter to start speaking, and press Enter again to end your answer for each question. Let's begin with your introduction. Can you tell me about yourself and your background in software engineering?"
        print("Interviewer:", welcome_message)
        await async_play_audio(client.tts(welcome_message, options))

        conversation_history = [
            {"role": "assistant", "content": welcome_message}
        ]

        while True:
            user_input = await async_get_speech_input()
            if user_input.lower() == 'quit':
                break
            
            conversation_history.append({"role": "user", "content": user_input})
            
            start_time = time.time()
            
            # Process candidate's response and generate next question
            relevant_info = get_relevant_information(user_input, docsearch)
            interviewer_response = generate_interview_response(groq_client, model, system_prompt, conversation_history, relevant_info)
            
            # Truncate the response if it's too long
            if len(interviewer_response) > 2000:
                interviewer_response = interviewer_response[:1997] + "..."
            
            conversation_history.append({"role": "assistant", "content": interviewer_response})
            
            print("Interviewer:", interviewer_response)
            await async_play_audio(client.tts(interviewer_response, options))
            
            print(f"Response time: {time.time() - start_time:.2f} seconds")

        # Closing message
        closing_message = "Thank you for participating in this interview. We've covered a range of topics, and I appreciate your responses. Do you have any questions about the role or the company?"
        print("Interviewer:", closing_message)
        await async_play_audio(client.tts(closing_message, options))

    except Exception as e:
        print(f"An error occurred during the interview: {str(e)}")

    finally:
        # Cleanup
        await client.close()
        audio_manager.terminate()

    return 0

# === ASYNC EXAMPLE ===


if __name__ == "__main__":
    asyncio.run(main())