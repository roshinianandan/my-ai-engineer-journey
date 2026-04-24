import os
import time
import tempfile
import threading


class TTSSynthesizer:
    """
    Text-to-Speech using pyttsx3.

    pyttsx3 works offline — no API calls needed.
    Uses the OS's built-in TTS engine:
    - Windows: SAPI5
    - macOS: NSSpeechSynthesizer
    - Linux: espeak

    For higher quality TTS, you could swap this for:
    - Coqui TTS (open source, better quality)
    - OpenAI TTS API (paid)
    - ElevenLabs (paid, very natural)
    """

    def __init__(
        self,
        rate: int = 175,        # words per minute
        volume: float = 1.0,    # 0.0 to 1.0
        voice_index: int = 0    # 0 = first available voice
    ):
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        """Initialize pyttsx3 TTS engine."""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", self.rate)
            self.engine.setProperty("volume", self.volume)

            # Set voice
            voices = self.engine.getProperty("voices")
            if voices and self.voice_index < len(voices):
                self.engine.setProperty(
                    "voice",
                    voices[self.voice_index].id
                )

            print(f"[TTS] Engine initialized | "
                  f"Rate: {self.rate} wpm | "
                  f"Volume: {self.volume}")

            if voices:
                print(f"[TTS] Voice: {voices[self.voice_index].name}")

        except ImportError:
            print("[TTS] pyttsx3 not installed: pip install pyttsx3")
            raise
        except Exception as e:
            print(f"[TTS] Init error: {e}")
            self.engine = None

    def speak(self, text: str, blocking: bool = True):
        """
        Convert text to speech and play it.
        blocking=True waits for speech to complete.
        blocking=False plays in background thread.
        """
        if not self.engine:
            print(f"[TTS] No engine — printing instead: {text}")
            return

        if not text.strip():
            return

        # Clean text for better speech
        clean_text = self._clean_for_speech(text)

        print(f"\n🔊 Speaking: '{clean_text[:60]}...'")

        if blocking:
            self.engine.say(clean_text)
            self.engine.runAndWait()
        else:
            thread = threading.Thread(
                target=self._speak_thread,
                args=(clean_text,)
            )
            thread.daemon = True
            thread.start()

    def _speak_thread(self, text: str):
        """Run TTS in a background thread."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTS Thread] Error: {e}")

    def _clean_for_speech(self, text: str) -> str:
        """Clean text for better TTS output."""
        import re

        # Remove markdown formatting
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)       # italic
        text = re.sub(r"`(.+?)`", r"\1", text)          # code
        text = re.sub(r"#{1,6}\s", "", text)             # headers
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text) # links

        # Replace special characters
        text = text.replace("→", "to")
        text = text.replace("←", "from")
        text = text.replace("≥", "greater than or equal to")
        text = text.replace("≤", "less than or equal to")

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def list_voices(self):
        """List all available TTS voices."""
        if not self.engine:
            print("No TTS engine available")
            return

        voices = self.engine.getProperty("voices")
        print(f"\n🎙️  Available TTS Voices ({len(voices)} total):")
        for i, voice in enumerate(voices):
            print(f"  [{i}] {voice.name}")
            print(f"      ID: {voice.id}")
            if hasattr(voice, "languages") and voice.languages:
                print(f"      Languages: {voice.languages}")

    def save_to_file(self, text: str, filepath: str = None) -> str:
        """Save TTS output to an audio file."""
        if not self.engine:
            return ""

        if filepath is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            filepath = tmp.name
            tmp.close()

        clean_text = self._clean_for_speech(text)
        self.engine.save_to_file(clean_text, filepath)
        self.engine.runAndWait()

        print(f"[TTS] Saved to: {filepath}")
        return filepath

    def set_rate(self, rate: int):
        """Change speech rate (words per minute)."""
        self.rate = rate
        if self.engine:
            self.engine.setProperty("rate", rate)

    def set_voice(self, index: int):
        """Change voice by index."""
        if not self.engine:
            return
        voices = self.engine.getProperty("voices")
        if 0 <= index < len(voices):
            self.engine.setProperty("voice", voices[index].id)
            print(f"[TTS] Voice set to: {voices[index].name}")