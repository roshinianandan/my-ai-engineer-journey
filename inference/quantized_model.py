import time
import os
from inference.gguf_loader import get_model_path, MODELS_DIR


class QuantizedModel:
    """
    Wrapper around llama-cpp-python for running GGUF quantized models.

    llama.cpp is a C++ inference engine optimized for running
    quantized LLMs on CPU and GPU. llama-cpp-python provides
    Python bindings so we can use it like any other Python library.

    Key advantages:
    - Runs 7B+ models on laptops with 8GB RAM
    - Much faster than HuggingFace on CPU
    - No GPU required for small models
    - Standard GGUF format works with any quantized model
    """

    def __init__(
        self,
        model_path: str = None,
        model_key: str = "tinyllama-q4",
        n_ctx: int = 2048,      # context window size
        n_threads: int = 4,     # CPU threads to use
        verbose: bool = False
    ):
        self.model_path = model_path or get_model_path(model_key)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.model = None
        self._load(verbose=verbose)

    def _load(self, verbose: bool = False):
        """Load the GGUF model using llama-cpp-python."""
        try:
            from llama_cpp import Llama

            print(f"[QuantizedModel] Loading: {os.path.basename(self.model_path)}")
            print(f"[QuantizedModel] Context: {self.n_ctx} tokens | "
                  f"Threads: {self.n_threads}")

            start = time.time()
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=verbose
            )
            elapsed = round(time.time() - start, 2)
            print(f"[QuantizedModel] Loaded in {elapsed}s")

        except ImportError:
            print("[QuantizedModel] llama-cpp-python not installed.")
            print("  Install with: pip install llama-cpp-python")
            raise

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: list = None
    ) -> dict:
        """
        Generate text from a prompt.
        Returns response dict with text and timing.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start = time.time()

        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or ["</s>", "\n\n\n"],
            echo=False
        )

        elapsed_ms = round((time.time() - start) * 1000, 2)
        generated_text = output["choices"][0]["text"].strip()
        tokens_generated = output["usage"]["completion_tokens"]
        tokens_per_sec = round(tokens_generated / (elapsed_ms / 1000), 1)

        return {
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "time_ms": elapsed_ms,
            "tokens_per_second": tokens_per_sec,
            "model": os.path.basename(self.model_path)
        }

    def chat(
        self,
        message: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> dict:
        """
        Chat with the model using a system + user prompt format.
        Formats the prompt in TinyLlama's chat template.
        """
        # TinyLlama chat format
        prompt = f"""<|system|>
{system_prompt}</s>
<|user|>
{message}</s>
<|assistant|>
"""
        return self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "<|user|>", "<|system|>"]
        )

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ):
        """Stream tokens one by one."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        print("🤖 Response: ", end="", flush=True)
        start = time.time()
        token_count = 0

        for token in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            echo=False
        ):
            text = token["choices"][0]["text"]
            print(text, end="", flush=True)
            token_count += 1

        elapsed = round(time.time() - start, 2)
        tps = round(token_count / elapsed, 1) if elapsed > 0 else 0
        print(f"\n\n   [{token_count} tokens | {elapsed}s | {tps} tok/s]")

    def get_info(self) -> dict:
        """Return model information."""
        size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        return {
            "model_path": self.model_path,
            "model_name": os.path.basename(self.model_path),
            "size_mb": round(size_mb, 1),
            "context_window": self.n_ctx,
            "threads": self.n_threads
        }