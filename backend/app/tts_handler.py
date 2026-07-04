import pyttsx3
import time


class TTSHandler:
    def __init__(self):
        pass

    def _get_engine(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        voices = engine.getProperty("voices")
        for voice in voices:
            if "male" in voice.name.lower() or "david" in voice.name.lower() or "mark" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
        return engine

    def speak(self, text: str):
        print("  [speaking...]")
        engine = self._get_engine()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        time.sleep(0.5)

    def close(self):
        pass


if __name__ == "__main__":
    tts = TTSHandler()
    test_text = "You mentioned a hash function maps to an index. What happens when two keys produce the same hash value?"
    print(f"Text: {test_text}\n")
    tts.speak(test_text)
    print("Done!")
    tts.close()