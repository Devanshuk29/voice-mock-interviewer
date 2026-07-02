import pyaudio
import wave
import os
import tempfile
import time
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

SILENCE_THRESHOLD = 500
SILENCE_TIMEOUT = 5


def is_silent(data: bytes) -> bool:
    import struct
    samples = struct.unpack(f"{len(data)//2}h", data)
    rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5
    return rms < SILENCE_THRESHOLD


class AudioHandler:
    def __init__(self):
        self.client = DeepgramClient(DEEPGRAM_API_KEY)
        self.audio = pyaudio.PyAudio()

    def record_answer(self) -> str:
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

        while True:
            data = stream.read(AUDIO_CONFIG["chunk"], exception_on_overflow=False)
            frames.append(data)

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

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(AUDIO_CONFIG["channels"])
            wf.setsampwidth(self.audio.get_sample_size(AUDIO_CONFIG["format"]))
            wf.setframerate(AUDIO_CONFIG["rate"])
            wf.writeframes(b"".join(frames))

        print("  [transcribing...]")

        with open(tmp_path, "rb") as audio_file:
            response = self.client.listen.rest.v("1").transcribe_file(
                {"buffer": audio_file},
                PrerecordedOptions(
                    model="nova-2",
                    language="en-IN",
                    smart_format=True,
                )
            )

        os.unlink(tmp_path)
        transcript = response.results.channels[0].alternatives[0].transcript
        return transcript

    def close(self):
        self.audio.terminate()


if __name__ == "__main__":
    print("Testing AudioHandler — speak freely, pause {SILENCE_TIMEOUT} seconds when done\n")
    handler = AudioHandler()
    transcript = handler.record_answer()
    print(f"\nFinal transcript: {transcript}")
    handler.close()
#A binary tree is a fundamental, non-linear hierarchical data structure consisting of nodes, 
# where each node has at most two children: a "left child" and a "right child". It is characterized 
# by a single topmost root node, with branches extending downward to child nodes and finally to 
# "leaf nodes" (nodes with no children).This specific two-child limit makes binary trees easier 
# to implement and navigate than general trees, allowing for faster searching, sorting, and data 
# storage
