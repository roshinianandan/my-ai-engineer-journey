import os
import time
import tempfile
import numpy as np


class WhisperTranscriber:
    """
    Speech-to-text using faster-whisper.

    faster-whisper is a reimplementation of Whisper that is
    2-4x faster than the original and uses less memory.

    Models available (download on first use):
    - tiny:   ~75MB, fastest, least accurate
    - base:   ~145MB, good balance
    - small:  ~465MB, better accuracy
    - medium: ~1.5GB, high accuracy
    - large:  ~3GB, best accuracy

    For voice interface, 'base' or 'small' is recommended.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            from faster_whisper import WhisperModel

            print(f"[Whisper] Loading {self.model_size} model "
                  f"(downloads on first use)...")
            start = time.time()

            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )

            elapsed = round(time.time() - start, 2)
            print(f"[Whisper] Model loaded in {elapsed}s")

        except ImportError:
            print("[Whisper] faster-whisper not installed.")
            print("  Install: pip install faster-whisper")
            raise

    def transcribe_file(
        self,
        audio_path: str,
        language: str = None,
        beam_size: int = 5
    ) -> dict:
        """
        Transcribe audio from a file.

        Returns dict with:
        - text: transcribed text
        - language: detected language
        - confidence: average confidence score
        - duration_ms: transcription time
        """
        if not os.path.exists(audio_path):
            return {"error": f"File not found: {audio_path}"}

        start = time.time()

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=beam_size,
            language=language,
            vad_filter=True,      # filter out silence
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        text_parts = []
        confidences = []

        for segment in segments:
            text_parts.append(segment.text.strip())
            # avg_logprob is log probability — convert to approximate confidence
            confidence = min(1.0, max(0.0, segment.avg_logprob + 1.0))
            confidences.append(confidence)

        full_text = " ".join(text_parts).strip()
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        elapsed_ms = round((time.time() - start) * 1000, 2)

        return {
            "text": full_text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "confidence": round(avg_confidence, 3),
            "duration_ms": elapsed_ms,
            "model": self.model_size
        }

    def transcribe_array(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = None
    ) -> dict:
        """Transcribe audio from a numpy array."""
        from voice.audio_utils import save_audio_to_file

        # Save to temp file
        temp_path = save_audio_to_file(audio, sample_rate)

        try:
            result = self.transcribe_file(temp_path, language=language)
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        return result

    def listen_and_transcribe(
        self,
        duration_seconds: int = 5,
        use_silence_detection: bool = True,
        language: str = None
    ) -> dict:
        """
        Record from microphone and transcribe.
        Full pipeline: record → save → transcribe.
        """
        from voice.audio_utils import (
            record_audio,
            record_until_silence,
            save_audio_to_file
        )

        if use_silence_detection:
            audio, sr = record_until_silence()
        else:
            audio, sr = record_audio(duration_seconds=duration_seconds)

        result = self.transcribe_array(audio, sr, language=language)

        print(f"\n📝 Transcribed: '{result.get('text', '')}'")
        if result.get("language"):
            print(f"   Language: {result['language']} "
                  f"({result.get('language_probability', 0):.0%})")
        print(f"   Time: {result.get('duration_ms', 0)}ms")

        return result