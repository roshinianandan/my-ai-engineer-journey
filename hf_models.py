from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import time

# ── MODEL OPTIONS ──────────────────────────────────────────────────────────
# These are small models that run on CPU or low-end GPU
# Change this to try different models from HuggingFace Hub
HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"   # Apple Silicon
    else:
        return "cpu"


def load_pipeline(model_name: str = HF_MODEL):
    """
    Load a HuggingFace text generation pipeline.
    The pipeline API handles tokenization, inference, and decoding automatically.
    First run downloads the model — subsequent runs load from cache.
    """
    device = get_device()
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print("Downloading model if not cached — this may take a few minutes...\n")

    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    print("Model loaded successfully!\n")
    return pipe


def generate(pipe, prompt: str, max_new_tokens: int = 200) -> dict:
    """
    Generate text using the HuggingFace pipeline.
    Returns the generated text and timing information.
    """
    start = time.time()

    # Format prompt for TinyLlama chat format
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]

    output = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=pipe.tokenizer.eos_token_id
    )

    elapsed = time.time() - start
    generated_text = output[0]["generated_text"][-1]["content"]

    return {
        "text": generated_text,
        "time_seconds": round(elapsed, 2),
        "model": HF_MODEL
    }


def explore_tokenizer(model_name: str = HF_MODEL):
    """
    Show how a tokenizer works — how text becomes numbers.
    This is one of the most important things to understand about LLMs.
    """
    print(f"\n{'='*55}")
    print("  TOKENIZER EXPLORER")
    print(f"{'='*55}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    examples = [
        "Hello world",
        "Machine learning is fascinating",
        "RAG stands for Retrieval Augmented Generation",
        "The quick brown fox jumps over the lazy dog"
    ]

    for text in examples:
        tokens = tokenizer.encode(text)
        decoded = [tokenizer.decode([t]) for t in tokens]
        print(f"Text:    {text}")
        print(f"Tokens:  {tokens}")
        print(f"Decoded: {decoded}")
        print(f"Count:   {len(tokens)} tokens\n")


def model_info(model_name: str = HF_MODEL):
    """Display information about a model."""
    print(f"\n{'='*55}")
    print(f"  MODEL INFO: {model_name}")
    print(f"{'='*55}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Vocabulary size:  {tokenizer.vocab_size:,} tokens")
    print(f"Model type:       {tokenizer.__class__.__name__}")
    print(f"Max length:       {tokenizer.model_max_length}")
    print(f"Padding token:    {tokenizer.pad_token}")
    print(f"EOS token:        {tokenizer.eos_token}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HuggingFace Model Explorer")
    parser.add_argument("--tokenize", action="store_true", help="Explore tokenizer")
    parser.add_argument("--info",     action="store_true", help="Show model info")
    parser.add_argument("--generate", type=str, help="Generate text for a prompt")
    args = parser.parse_args()

    if args.tokenize:
        explore_tokenizer()
    elif args.info:
        model_info()
    elif args.generate:
        pipe = load_pipeline()
        result = generate(pipe, args.generate)
        print(f"\n🤖 Response: {result['text']}")
        print(f"   Time: {result['time_seconds']}s")
    else:
        # Default: show tokenizer and generate a sample
        explore_tokenizer()
        pipe = load_pipeline()
        result = generate(pipe, "What is machine learning? Explain in 3 sentences.")
        print(f"\n🤖 Response: {result['text']}")
        print(f"   Time: {result['time_seconds']}s")