import os
import json
import argparse
from datetime import datetime
import ollama
from config import MODEL, TEMPERATURE, MAX_TOKENS
from memory.memory_manager import MemoryManager

PERSONAS = {
    "default":  "You are a helpful AI assistant for an AIML student. Be concise and use examples when explaining concepts.",
    "mentor":   "You are a senior ML engineer mentoring a student. Be encouraging, technical, and suggest resources.",
    "socratic": "You are a Socratic teacher. Never give direct answers — guide through questions.",
    "pirate":   "You are a pirate who became an ML expert. Use nautical metaphors. Arrr."
}


def save_log(history: list, persona: str):
    os.makedirs("chat_logs", exist_ok=True)
    filename = f"chat_logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{persona}.json"
    with open(filename, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n💾 Conversation saved to {filename}")


def chat(persona: str = "default", user_id: str = "roshini"):
    system_prompt = PERSONAS.get(persona, PERSONAS["default"])

    # Initialise memory manager — combines short + long term memory
    memory = MemoryManager(
        user_id=user_id,
        system_prompt=system_prompt,
        max_buffer=10
    )

    print(f"\n🤖 AI Assistant  |  Model: [{MODEL}]  |  Persona: [{persona}]")
    print(f"   User: {user_id} | Memory: ON")
    print("   Commands: 'save' 'clear' 'memories' 'forget <word>' 'stats' 'quit'\n")
    print("─" * 55)

    session_log = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            save_log(session_log, persona)
            print("👋 Goodbye!")
            break

        if user_input.lower() == "save":
            save_log(session_log, persona)
            continue

        if user_input.lower() == "clear":
            memory.short_term.clear()
            print("🔄 Short-term memory cleared.\n")
            continue

        if user_input.lower() == "memories":
            memory.show_memories()
            continue

        if user_input.lower() == "stats":
            memory.stats()
            continue

        if user_input.lower().startswith("forget "):
            keyword = user_input[7:].strip()
            memory.forget(keyword)
            continue

        if user_input.lower().startswith("remember "):
            fact = user_input[9:].strip()
            memory.save_memory(fact, category="manual")
            continue

        # Process input through memory manager
        messages = memory.process_input(user_input)

        try:
            print("\n🤖 Assistant: ", end="", flush=True)
            full_reply = ""

            for chunk in ollama.chat(
                model=MODEL,
                messages=messages,
                stream=True,
                options={"temperature": TEMPERATURE,
                         "num_predict": MAX_TOKENS}
            ):
                token = chunk["message"]["content"]
                print(token, end="", flush=True)
                full_reply += token

            print()
            memory.process_response(full_reply)
            session_log.append({
                "user": user_input,
                "assistant": full_reply
            })

        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", choices=list(PERSONAS.keys()), default="default")
    parser.add_argument("--user",    type=str, default="roshini",
                        help="User ID for memory isolation")
    args = parser.parse_args()
    chat(persona=args.persona, user_id=args.user)