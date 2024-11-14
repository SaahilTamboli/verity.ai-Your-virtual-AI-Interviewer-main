import datetime
import time

import threading
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from dotenv import load_dotenv

load_dotenv()

class DeepgramSTT:
    def __init__(self):
        self.full_transcription = ""  # Full transcript storage
        self.final_transcription = ""
        self.other_text= ""
        self.transcript_ready = threading.Event()  # Event to signal that the final transcript is ready
        self.connection_status = False  # Track whether the connection is active
        self.deepgram = DeepgramClient()
        self.connection = self.deepgram.listen.live.v("1")
        self.setup_events()
        self.start_time = 0
        self.transcription_time = 0

    def setup_events(self):
        self.connection.on(LiveTranscriptionEvents.Open, self.on_open)
        self.connection.on(LiveTranscriptionEvents.Close, self.on_close)
        self.connection.on(LiveTranscriptionEvents.Transcript, self.on_message)
        self.connection.on(LiveTranscriptionEvents.SpeechStarted, self.on_speech_started)

    def on_open(self, *args, **kwargs):
        self.connection_status = True
        print("Connection opened")
    def on_speech_started(self, x, speech_started, **kwargs):
        print("Listening...")
    def on_close(self, *args, **kwargs):
        self.connection_status = False
        print("Connection closed")

    def on_message(self, x, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if len(sentence) == 0:
            return
        if result.is_final and result.speech_final:
            self.final_transcription = self.full_transcription + sentence
            self.transcription_time = (time.time() - self.start_time) * 1000  # Convert to milliseconds
            self.transcript_ready.set()
            self.start_time = time.time()  # Reset start time for next transcription
            return
        elif result.is_final and not result.speech_final: 
            self.full_transcription += sentence + " "        
            
            return
        else:
            self.other_text=sentence

    def start_connection(self):
        self.start_time = time.time()
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            punctuate=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            # interim_results=True,
            # utterance_end_ms="1000",
            vad_events=True,
            endpointing=100
        )
        if not self.connection.start(options):
            print("Failed to start connection")
            return False
        return True

    def send_audio_data(self, data):
        self.connection.send(data)

    def finish(self):
        self.connection.finish()
        print("Finished")
        self.print_final_transcript()

    def print_final_transcript(self):
        print("Complete final transcript:")
        print(self.full_transcription)

    def is_connection_active(self):
        return self.connection_status
