import time
import wave
import tempfile
import os
import numpy as np


def record_audio(
    duration_seconds: int = 5,
    sample_rate: int = 16000,
    channels: int = 1
) -> tuple:
    """
    Record audio from the default microphone.

    Returns (audio_data_numpy_array, sample_rate)

    sample_rate=16000 is required by Whisper.
    channels=1 (mono) is required by Whisper.
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError("Install sounddevice: pip install sounddevice")

    print(f"🎤 Recording for {duration_seconds}s... Speak now!")

    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype=np.float32
    )
    sd.wait()  # Wait until recording is complete
    print("✅ Recording complete")

    return audio.flatten(), sample_rate


def record_until_silence(
    silence_threshold: float = 0.01,
    silence_duration: float = 1.5,
    max_duration: int = 30,
    sample_rate: int = 16000
) -> tuple:
    """
    Record audio until silence is detected.
    Automatically stops when the user stops speaking.

    silence_threshold: RMS amplitude below which is considered silence
    silence_duration: seconds of silence before stopping
    max_duration: maximum recording time in seconds
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError("Install sounddevice: pip install sounddevice")

    print("🎤 Listening... (speak now, recording stops on silence)")

    chunk_size = int(sample_rate * 0.1)  # 100ms chunks
    max_chunks = int(max_duration * sample_rate / chunk_size)

    audio_chunks = []
    silent_chunks = 0
    silent_chunks_threshold = int(silence_duration / 0.1)
    speaking_started = False

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        blocksize=chunk_size
    ) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            chunk = chunk.flatten()
            audio_chunks.append(chunk)

            # Check RMS amplitude
            rms = np.sqrt(np.mean(chunk ** 2))

            if rms > silence_threshold:
                speaking_started = True
                silent_chunks = 0
                print(".", end="", flush=True)
            else:
                if speaking_started:
                    silent_chunks += 1
                    if silent_chunks >= silent_chunks_threshold:
                        print("\n✅ Silence detected — stopping")
                        break

    audio = np.concatenate(audio_chunks)
    return audio, sample_rate


def save_audio_to_file(
    audio: np.ndarray,
    sample_rate: int = 16000,
    filepath: str = None
) -> str:
    """Save audio array to a WAV file."""
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Install soundfile: pip install soundfile")

    if filepath is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        filepath = tmp.name
        tmp.close()

    # Ensure audio is in the right format
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    sf.write(filepath, audio, sample_rate)
    return filepath


def play_audio_file(filepath: str):
    """Play an audio file."""
    try:
        import sounddevice as sd
        import soundfile as sf

        data, sample_rate = sf.read(filepath)
        print(f"🔊 Playing audio...")
        sd.play(data, sample_rate)
        sd.wait()

    except Exception as e:
        print(f"[Audio] Could not play audio: {e}")


def list_audio_devices():
    """List available audio input/output devices."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("\n🎤 Available Audio Devices:")
        for i, device in enumerate(devices):
            input_ch = device['max_input_channels']
            output_ch = device['max_output_channels']
            if input_ch > 0 or output_ch > 0:
                io = []
                if input_ch > 0:
                    io.append(f"IN:{input_ch}ch")
                if output_ch > 0:
                    io.append(f"OUT:{output_ch}ch")
                print(f"  [{i}] {device['name']} ({', '.join(io)})")
    except Exception as e:
        print(f"Could not list audio devices: {e}")