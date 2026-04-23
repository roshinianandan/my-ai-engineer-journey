import os
import time
from pathlib import Path
from huggingface_hub import hf_hub_download


MODELS_DIR = "./models"

# Available GGUF models — small enough to download quickly
AVAILABLE_MODELS = {
    "tinyllama-q4": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_gb": 0.67,
        "quantization": "Q4_K_M",
        "description": "TinyLlama 1.1B — 4-bit quantized, fast on CPU"
    },
    "tinyllama-q8": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        "size_gb": 1.1,
        "quantization": "Q8_0",
        "description": "TinyLlama 1.1B — 8-bit quantized, better quality"
    },
    "phi2-q4": {
        "repo_id": "TheBloke/phi-2-GGUF",
        "filename": "phi-2.Q4_K_M.gguf",
        "size_gb": 1.6,
        "quantization": "Q4_K_M",
        "description": "Microsoft Phi-2 2.7B — 4-bit, excellent for its size"
    }
}


def list_available_models():
    """List all available GGUF models for download."""
    print(f"\n{'='*60}")
    print("  AVAILABLE GGUF MODELS")
    print(f"{'='*60}")
    for key, info in AVAILABLE_MODELS.items():
        print(f"\n  {key}:")
        print(f"    Description:  {info['description']}")
        print(f"    Size:         {info['size_gb']}GB")
        print(f"    Quantization: {info['quantization']}")
        print(f"    Repo:         {info['repo_id']}")
    print(f"{'='*60}\n")


def list_downloaded_models() -> list:
    """List GGUF models already downloaded to the models/ folder."""
    models_path = Path(MODELS_DIR)
    if not models_path.exists():
        return []

    gguf_files = list(models_path.glob("*.gguf"))
    if gguf_files:
        print(f"\n📁 Downloaded models in {MODELS_DIR}/:")
        for f in gguf_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   {f.name} ({size_mb:.0f} MB)")
    else:
        print(f"\n📁 No GGUF models downloaded yet in {MODELS_DIR}/")

    return gguf_files


def download_model(model_key: str) -> str:
    """
    Download a GGUF model from HuggingFace Hub.
    Returns the local path to the downloaded model.
    """
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_key}. "
            f"Available: {list(AVAILABLE_MODELS.keys())}"
        )

    model_info = AVAILABLE_MODELS[model_key]
    local_path = os.path.join(MODELS_DIR, model_info["filename"])

    # Check if already downloaded
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"[GGUFLoader] Model already downloaded: {local_path} ({size_mb:.0f}MB)")
        return local_path

    print(f"[GGUFLoader] Downloading {model_key}...")
    print(f"  Repository: {model_info['repo_id']}")
    print(f"  File:       {model_info['filename']}")
    print(f"  Size:       ~{model_info['size_gb']}GB")
    print("  This may take several minutes...\n")

    os.makedirs(MODELS_DIR, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=model_info["repo_id"],
        filename=model_info["filename"],
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False
    )

    print(f"\n[GGUFLoader] Download complete: {downloaded_path}")
    return downloaded_path


def get_model_path(model_key: str = None, model_path: str = None) -> str:
    """
    Get the path to a GGUF model.
    Either by model_key (downloads if needed) or direct path.
    """
    if model_path and os.path.exists(model_path):
        return model_path

    if model_key:
        return download_model(model_key)

    # Try to find any downloaded model
    downloaded = list_downloaded_models()
    if downloaded:
        return str(downloaded[0])

    raise FileNotFoundError(
        "No GGUF model found. Run with --download tinyllama-q4 first."
    )