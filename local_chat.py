import time
from hf_models import load_pipeline, generate, HF_MODEL


def chat_hf():
    """
    Interactive chat using a local HuggingFace model.
    Maintains conversation history just like chat.py but uses
    a HuggingFace model instead of Ollama.
    """
    print(f"\n🤖 HuggingFace Chat — Model: {HF_MODEL}")
    print("   Loading model... (first run downloads it)\n")

    pipe = load_pipeline()

    print("Model ready! Type 'quit' to exit.\n")
    print("-" * 55)

    conversation = []

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

        # Build message history
        messages = [{"role": "system",
                     "content": "You are a helpful AI assistant for an AIML student."}]
        messages.extend(conversation)
        messages.append({"role": "user", "content": user_input})

        print("\n🤖 Assistant: ", end="", flush=True)
        start = time.time()

        output = pipe(
            messages,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            pad_token_id=pipe.tokenizer.eos_token_id
        )

        reply = output[0]["generated_text"][-1]["content"]
        elapsed = round(time.time() - start, 2)

        print(reply)
        print(f"\n   [{elapsed}s]")

        # Add to history
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    chat_hf()