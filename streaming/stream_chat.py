import sys
import time
import ollama
import argparse
from config import MODEL, TEMPERATURE, MAX_TOKENS

PERSONAS = {
    "default":  "You are a helpful AI assistant for an AIML student. Be concise and use examples.",
    "mentor":   "You are a senior ML engineer mentoring a student. Be encouraging and technical.",
    "socratic": "You are a Socratic teacher. Guide through questions, never give direct answers.",
    "pirate":   "You are a pirate who became an ML expert. Use nautical metaphors. Arrr."
}


def stream_response(
    prompt: str,
    system_prompt: str = "",
    history: list = None,
    show_stats: bool = True
) -> str:
    """
    Stream a response token by token.
    Returns the full response string when complete.

    This is how every modern AI chat interface works —
    tokens arrive one at a time and are printed immediately
    rather than waiting for the full response.
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": prompt})

    start_time = time.time()
    full_reply = ""
    token_count = 0

    print("\n🤖 Assistant: ", end="", flush=True)

    for chunk in ollama.chat(
        model=MODEL,
        messages=messages,
        stream=True,
        options={
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS
        }
    ):
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_reply += token
        token_count += 1

    elapsed = round(time.time() - start_time, 2)
    tokens_per_sec = round(token_count / elapsed, 1) if elapsed > 0 else 0

    if show_stats:
        print(f"\n\n   [{elapsed}s | ~{token_count} tokens | {tokens_per_sec} tok/s]")
    else:
        print()

    return full_reply


def stream_with_thinking(prompt: str, system_prompt: str = "") -> str:
    """
    Stream response with a thinking indicator.
    Shows 'Thinking...' animation while the first token arrives,
    then switches to streaming the actual response.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    print("\n🤖 Thinking", end="", flush=True)

    stream = ollama.chat(
        model=MODEL,
        messages=messages,
        stream=True,
        options={"temperature": TEMPERATURE}
    )

    full_reply = ""
    first_token = True

    for chunk in stream:
        token = chunk["message"]["content"]

        if first_token:
            print(f"\r🤖 Assistant: {token}", end="", flush=True)
            first_token = False
        else:
            print(token, end="", flush=True)

        full_reply += token

    print()
    return full_reply


def interactive_streaming_chat(persona: str = "default"):
    """
    Full streaming chat session with conversation history.
    Every response streams token by token.
    """
    system_prompt = PERSONAS.get(persona, PERSONAS["default"])

    print(f"\n🤖 Streaming Chat  |  Model: [{MODEL}]  |  Persona: [{persona}]")
    print("   Type 'quit' to exit  |  'clear' to reset  |  'stats' to toggle stats\n")
    print("─" * 55)

    history = []
    show_stats = True

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break

        if user_input.lower() == "clear":
            history = []
            print("🔄 Conversation cleared.\n")
            continue

        if user_input.lower() == "stats":
            show_stats = not show_stats
            print(f"Stats: {'ON' if show_stats else 'OFF'}")
            continue

        reply = stream_response(
            prompt=user_input,
            system_prompt=system_prompt,
            history=history,
            show_stats=show_stats
        )

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})


def benchmark_streaming(prompts: list = None) -> dict:
    """
    Benchmark streaming speed on a set of prompts.
    Measures tokens per second — useful for comparing models.
    """
    if not prompts:
        prompts = [
            "What is machine learning? Answer in 2 sentences.",
            "List 5 popular Python libraries for data science.",
            "Explain gradient descent in simple terms."
        ]

    print(f"\n{'='*55}")
    print("  STREAMING BENCHMARK")
    print(f"  Model: {MODEL}")
    print(f"{'='*55}\n")

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt[:50]}...")

        start = time.time()
        full_reply = ""
        token_count = 0

        for chunk in ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": 0.3}
        ):
            token = chunk["message"]["content"]
            full_reply += token
            token_count += 1
            sys.stdout.write(".")
            sys.stdout.flush()

        elapsed = round(time.time() - start, 2)
        tps = round(token_count / elapsed, 1) if elapsed > 0 else 0

        print(f"\n   Time: {elapsed}s | Tokens: {token_count} | Speed: {tps} tok/s\n")
        results.append({
            "prompt": prompt[:50],
            "time": elapsed,
            "tokens": token_count,
            "tokens_per_second": tps
        })

    avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
    print(f"{'='*55}")
    print(f"  Average speed: {avg_tps:.1f} tokens/second")
    print(f"{'='*55}\n")

    return {"results": results, "avg_tokens_per_second": avg_tps}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Chat")
    parser.add_argument("--persona",    type=str, default="default",
                        choices=list(PERSONAS.keys()))
    parser.add_argument("--benchmark",  action="store_true",
                        help="Run speed benchmark")
    parser.add_argument("--thinking",   action="store_true",
                        help="Use thinking indicator mode")
    parser.add_argument("--prompt",     type=str,
                        help="Stream a single prompt")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_streaming()
    elif args.thinking and args.prompt:
        stream_with_thinking(args.prompt)
    elif args.prompt:
        stream_response(args.prompt)
    else:
        interactive_streaming_chat(persona=args.persona)