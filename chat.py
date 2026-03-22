import os
import json
import argparse
from datetime import datetime
import ollama
from config import MODEL, TEMPERATURE, MAX_TOKENS

PERSONAS = {
    "default":  "You are a helpful AI assistant for an AIML student. Be concise and use examples when explaining concepts.",
    "mentor":   "You are a senior ML engineer mentoring a student. Be encouraging, technical, and suggest resources.",
    "socratic": "You are a Socratic teacher. Never give direct answers — guide the student to the answer through questions.",
    "pirate":   "You are a pirate who somehow became an ML expert. Explain everything with nautical metaphors. Arrr."
}

def save_log(history: list, persona: str):
    os.makedirs("chat_logs", exist_ok=True)
    filename = f"chat_logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{persona}.json"
    with open(filename, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n💾 Conversation saved to {filename}")

def chat(persona: str = "default"):
    system_prompt = PERSONAS.get(persona, PERSONAS["default"])

    print(f"\n🤖 AI Assistant ready  |  Model: [{MODEL}]  |  Persona: [{persona}]")
    print("   Type 'quit' to exit  |  'save' to save log  |  'clear' to reset memory\n")
    print("─" * 55)

    conversation_history = [{"role": "system", "content": system_prompt}]
    session_chars = 0

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Interrupted. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye"]:
            save_log(conversation_history, persona)
            print("👋 Goodbye!")
            break

        if user_input.lower() == "save":
            save_log(conversation_history, persona)
            continue

        if user_input.lower() == "clear":
            conversation_history = [{"role": "system", "content": system_prompt}]
            session_chars = 0
            print("🔄 Memory cleared. Starting fresh.\n")
            continue

        conversation_history.append({"role": "user", "content": user_input})

        try:
            print("\n🤖 Assistant: ", end="", flush=True)

            full_reply = ""
            for chunk in ollama.chat(
                model=MODEL,
                messages=conversation_history,
                stream=True,
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_TOKENS
                }
            ):
                token = chunk["message"]["content"]
                print(token, end="", flush=True)
                full_reply += token

            print()

            session_chars += len(full_reply)
            conversation_history.append({"role": "assistant", "content": full_reply})
            print(f"\n   [Response chars: {len(full_reply)} | Session total: {session_chars}]")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Make sure Ollama is running — open a new terminal and run: ollama serve")
            conversation_history.pop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Assistant CLI")
    parser.add_argument(
        "--persona",
        choices=list(PERSONAS.keys()),
        default="default",
        help="Choose the assistant's personality"
    )
    args = parser.parse_args()
    chat(persona=args.persona)