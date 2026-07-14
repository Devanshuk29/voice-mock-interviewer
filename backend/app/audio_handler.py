import pyaudio
import wave
import os
import tempfile
import time
import struct
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

AUDIO_CONFIG = {
    "format": pyaudio.paInt16,
    "channels": 1,
    "rate": 16000,
    "chunk": 1024,
}

SILENCE_THRESHOLD = 800
SILENCE_TIMEOUT = 5
MAX_WAIT = 15


def is_silent(data: bytes) -> bool:
    samples = struct.unpack(f"{len(data)//2}h", data)
    rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5
    return rms < SILENCE_THRESHOLD


class AudioHandler:
    def __init__(self):
        self.client = DeepgramClient(DEEPGRAM_API_KEY)
        self.audio = pyaudio.PyAudio()

    def record_answer(self) -> str:
        try:
            stream = self.audio.open(
                format=AUDIO_CONFIG["format"],
                channels=AUDIO_CONFIG["channels"],
                rate=AUDIO_CONFIG["rate"],
                input=True,
                frames_per_buffer=AUDIO_CONFIG["chunk"],
            )

            print(f"  [listening... pause for {SILENCE_TIMEOUT}s when done speaking]")

            frames = []
            silence_start = None
            speech_detected = False
            start_time = time.time()

            while True:
                data = stream.read(AUDIO_CONFIG["chunk"], exception_on_overflow=False)
                frames.append(data)

                if not speech_detected and time.time() - start_time > MAX_WAIT:
                    print("  [no speech detected, timing out]")
                    break

                if is_silent(data):
                    if speech_detected:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > SILENCE_TIMEOUT:
                            print("  [silence detected, done recording]")
                            break
                else:
                    speech_detected = True
                    silence_start = None

            stream.stop_stream()
            stream.close()

            if not speech_detected:
                return ""

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(AUDIO_CONFIG["channels"])
                wf.setsampwidth(self.audio.get_sample_size(AUDIO_CONFIG["format"]))
                wf.setframerate(AUDIO_CONFIG["rate"])
                wf.writeframes(b"".join(frames))

            print("  [transcribing...]")

            try:
                with open(tmp_path, "rb") as audio_file:
                    response = self.client.listen.rest.v("1").transcribe_file(
                        {"buffer": audio_file},
                        PrerecordedOptions(
                            model="nova-2",
                            language="en-IN",
                            smart_format=True,
                        )
                    )
                transcript = response.results.channels[0].alternatives[0].transcript

                if len(transcript.strip()) < 3:
                    return ""

                return transcript

            except Exception as e:
                print(f"  [transcription error: {e}]")
                return ""
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            print(f"  [audio error: {e}]")
            return ""

    def close(self):
        self.audio.terminate()


if __name__ == "__main__":
    print(f"Testing AudioHandler — speak freely, pause {SILENCE_TIMEOUT} seconds when done\n")
    handler = AudioHandler()
    transcript = handler.record_answer()
    print(f"\nFinal transcript: {transcript}")
    handler.close()