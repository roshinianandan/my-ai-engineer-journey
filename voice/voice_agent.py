import time
import ollama
import argparse
from config import MODEL, TEMPERATURE, MAX_TOKENS

PERSONAS = {
    "default": "You are a helpful AI assistant. Keep responses concise — under 3 sentences when possible, since your responses will be spoken aloud.",
    "mentor": "You are a senior ML engineer mentoring a student. Be encouraging and brief — your responses are spoken aloud.",
    "friendly": "You are a friendly conversational AI. Speak naturally and warmly, keep responses short and conversational."
}


class VoiceAgent:
    """
    Full speech-to-speech AI assistant.

    Pipeline:
    [Microphone] → [Whisper STT] → [LLM] → [TTS] → [Speaker]

    This is the complete voice interface that transforms the
    CLI chatbot into a voice-driven AI assistant.
    """

    def __init__(
        self,
        persona: str = "default",
        whisper_model: str = "base",
        tts_rate: int = 175,
        use_silence_detection: bool = True,
        recording_duration: int = 5,
        verbose: bool = True
    ):
        self.persona = persona
        self.use_silence_detection = use_silence_detection
        self.recording_duration = recording_duration
        self.verbose = verbose
        self.conversation_history = []
        self.system_prompt = PERSONAS.get(persona, PERSONAS["default"])

        print(f"\n🤖 Initializing Voice Agent...")
        print(f"   Model:   {MODEL}")
        print(f"   Persona: {persona}")
        print(f"   Whisper: {whisper_model}")

        # Initialize transcriber
        print("\n[1/2] Loading Whisper transcriber...")
        from voice.transcriber import WhisperTranscriber
        self.transcriber = WhisperTranscriber(model_size=whisper_model)

        # Initialize synthesizer
        print("[2/2] Loading TTS synthesizer...")
        from voice.synthesizer import TTSSynthesizer
        self.synthesizer = TTSSynthesizer(rate=tts_rate)

        print("\n✅ Voice Agent ready!\n")

    def _get_llm_response(self, user_text: str) -> str:
        """Get LLM response for transcribed text."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history[-10:])
        messages.append({"role": "user", "content": user_text})

        response = ollama.chat(
            model=MODEL,
            messages=messages,
            stream=False,
            options={
                "temperature": TEMPERATURE,
                "num_predict": 200  # shorter for voice
            }
        )

        return response["message"]["content"]

    def process_voice_input(self) -> dict:
        """
        Single voice interaction cycle:
        Record → Transcribe → LLM → Speak
        """
        # Step 1: Record
        print("\n" + "="*50)
        result = self.transcriber.listen_and_transcribe(
            duration_seconds=self.recording_duration,
            use_silence_detection=self.use_silence_detection
        )

        user_text = result.get("text", "").strip()

        if not user_text:
            self.synthesizer.speak("I didn't catch that. Please try again.")
            return {"success": False, "reason": "empty_transcription"}

        print(f"\n🗣️  You said: '{user_text}'")

        # Step 2: Check for commands
        if any(word in user_text.lower()
               for word in ["goodbye", "bye", "exit", "quit", "stop"]):
            self.synthesizer.speak(
                "Goodbye! It was great talking with you."
            )
            return {"success": True, "command": "exit"}

        if "clear history" in user_text.lower():
            self.conversation_history = []
            self.synthesizer.speak(
                "Conversation history cleared. Starting fresh!"
            )
            return {"success": True, "command": "clear"}

        # Step 3: Get LLM response
        print("🧠 Thinking...")
        start = time.time()
        response_text = self._get_llm_response(user_text)
        llm_time = round(time.time() - start, 2)

        print(f"\n🤖 Response ({llm_time}s): {response_text[:100]}...")

        # Step 4: Update history
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })

        # Step 5: Speak response
        self.synthesizer.speak(response_text)

        return {
            "success": True,
            "user_text": user_text,
            "response": response_text,
            "llm_time": llm_time,
            "language": result.get("language", "en")
        }

    def run(self, max_turns: int = None):
        """
        Run the voice agent in a continuous loop.
        max_turns: stop after N turns (None = run indefinitely)
        """
        print(f"🎙️  Voice Agent Running — {self.persona} mode")
        print("   Say 'goodbye' or 'exit' to stop")
        print("   Say 'clear history' to reset conversation")
        print("="*50)

        # Greet the user
        self.synthesizer.speak(
            f"Hello! I am your AI assistant in {self.persona} mode. "
            "How can I help you today?"
        )

        turn = 0
        while True:
            if max_turns and turn >= max_turns:
                print(f"\nReached max turns ({max_turns}). Stopping.")
                break

            try:
                result = self.process_voice_input()
                turn += 1

                if result.get("command") == "exit":
                    break

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user")
                self.synthesizer.speak("Goodbye!")
                break

            except Exception as e:
                print(f"\n❌ Error: {e}")
                try:
                    self.synthesizer.speak(
                        "I encountered an error. Please try again."
                    )
                except Exception:
                    pass

        print(f"\n[Voice Agent] Session complete — {turn} turns")

    def text_to_speech_demo(self, text: str = None):
        """Demo TTS with sample text."""
        sample = text or (
            "Hello! I am your AI assistant. "
            "I can understand your voice and respond naturally. "
            "This is powered by Whisper for speech recognition "
            "and a local language model for responses."
        )
        print(f"\n🔊 TTS Demo: '{sample[:60]}...'")
        self.synthesizer.speak(sample)

    def speech_to_text_demo(self, duration: int = 5):
        """Demo STT — record and transcribe."""
        print(f"\n🎤 STT Demo — recording {duration}s")
        result = self.transcriber.listen_and_transcribe(
            duration_seconds=duration,
            use_silence_detection=False
        )
        print(f"\nTranscription result: {result}")
        return result


def run_tts_only(text: str, rate: int = 175):
    """Quick TTS without full agent init."""
    from voice.synthesizer import TTSSynthesizer
    tts = TTSSynthesizer(rate=rate)
    tts.speak(text)


def run_stt_only(duration: int = 5, model: str = "base"):
    """Quick STT without full agent init."""
    from voice.transcriber import WhisperTranscriber
    transcriber = WhisperTranscriber(model_size=model)
    result = transcriber.listen_and_transcribe(
        duration_seconds=duration,
        use_silence_detection=True
    )
    print(f"\nResult: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice AI Assistant")
    parser.add_argument("--run",      action="store_true",
                        help="Start full voice agent")
    parser.add_argument("--persona",  type=str, default="default",
                        choices=list(PERSONAS.keys()))
    parser.add_argument("--whisper",  type=str, default="base",
                        choices=["tiny", "base", "small"],
                        help="Whisper model size")
    parser.add_argument("--tts",      type=str,
                        help="Test TTS with text")
    parser.add_argument("--stt",      action="store_true",
                        help="Test STT — record and transcribe")
    parser.add_argument("--duration", type=int, default=5,
                        help="Recording duration in seconds")
    parser.add_argument("--voices",   action="store_true",
                        help="List available TTS voices")
    parser.add_argument("--devices",  action="store_true",
                        help="List audio devices")
    parser.add_argument("--turns",    type=int, default=None,
                        help="Max conversation turns")
    parser.add_argument("--rate",     type=int, default=175,
                        help="TTS speech rate (words per minute)")
    args = parser.parse_args()

    if args.devices:
        from voice.audio_utils import list_audio_devices
        list_audio_devices()

    elif args.voices:
        from voice.synthesizer import TTSSynthesizer
        tts = TTSSynthesizer()
        tts.list_voices()

    elif args.tts:
        run_tts_only(args.tts, rate=args.rate)

    elif args.stt:
        run_stt_only(duration=args.duration, model=args.whisper)

    elif args.run:
        agent = VoiceAgent(
            persona=args.persona,
            whisper_model=args.whisper,
            tts_rate=args.rate
        )
        agent.run(max_turns=args.turns)

    else:
        # Default: demo both TTS and STT
        print("\n🎙️  Voice Interface Demo")
        print("   Testing TTS first, then STT...\n")

        print("--- TTS Test ---")
        run_tts_only(
            "Hello! Voice interface is working. "
            "I can speak responses to you.",
            rate=args.rate
        )

        print("\n--- STT Test ---")
        print("Recording 5 seconds of audio...")
        run_stt_only(duration=5, model=args.whisper)