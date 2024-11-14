from deepgramstt import DeepgramSTT
from datetime import datetime
import threading
import pyaudio

def main():
    # Audio stream configuration
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = 16000
    FRAMES_PER_BUFFER = 3200
    
    
    
    

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)
    except IOError as e:
        print(f"Could not open audio stream: {e}")
        p.terminate()
        return

    # Initialize DeepgramSTT
    dg_connection = DeepgramSTT()
    if not dg_connection.start_connection():
        print("Failed to start Deepgram connection")
        stream.stop_stream()
        stream.close()
        p.terminate()
        return
    
    print("Connection started. Begin speaking now.")
    
    # Start the audio stream thread immediately
    exit_flag = False

    def audio_stream_thread():
  
        try:
            while not exit_flag and dg_connection.is_connection_active():
                try:
                    data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                except IOError as e:
                    print(f"Error reading audio data: {e}")
                    break  # Exit the loop if we can't read the data
                dg_connection.send_audio_data(data)

                if dg_connection.transcript_ready.is_set():  # Non-blocking check for the event
                    print(f"Transcription: {dg_connection.final_transcription}")
                    print(f"Latency: {dg_connection.transcription_time:.2f} ms")
                    dg_connection.final_transcription = ""
                    dg_connection.transcript_ready.clear()  # Reset the event
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            dg_connection.finish()

    audio_thread = threading.Thread(target=audio_stream_thread)
    audio_thread.start()

    input("Press Enter to stop recording...\n")
    exit_flag = True
    audio_thread.join()

    print("Finished recording and processing.")

if __name__ == "__main__":
    main()
