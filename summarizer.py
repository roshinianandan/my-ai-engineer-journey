import argparse
import ollama
from prompts.templates import (
    ZERO_SHOT_SUMMARY,
    FEW_SHOT_SUMMARY,
    CHAIN_OF_THOUGHT_SUMMARY,
    STYLE_PROMPTS
)
from config import MODEL, TEMPERATURE

def call_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the response."""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": TEMPERATURE}
    )
    return response["message"]["content"]

def summarize(text: str, mode: str = "zero", style: str = None) -> str:
    """
    Summarize text using different prompting strategies.
    mode: zero | few | cot
    style: eli5 | technical | executive
    """
    if style:
        if style not in STYLE_PROMPTS:
            print(f"Unknown style '{style}'. Choose from: eli5, technical, executive")
            return ""
        prompt = STYLE_PROMPTS[style].format(text=text)
        label = f"Style: {style.upper()}"

    elif mode == "zero":
        prompt = ZERO_SHOT_SUMMARY.format(text=text)
        label = "Mode: ZERO-SHOT"

    elif mode == "few":
        prompt = FEW_SHOT_SUMMARY.format(text=text)
        label = "Mode: FEW-SHOT"

    elif mode == "cot":
        prompt = CHAIN_OF_THOUGHT_SUMMARY.format(text=text)
        label = "Mode: CHAIN-OF-THOUGHT"

    else:
        print(f"Unknown mode '{mode}'. Choose from: zero, few, cot")
        return ""

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}\n")

    result = call_ollama(prompt)
    print(result)
    return result

def compare_all(text: str):
    """Run all three prompting strategies and show results side by side."""
    print("\n🔬 COMPARING ALL PROMPTING STRATEGIES")
    print("=" * 55)
    print("Input text:", text[:100], "...\n" if len(text) > 100 else "\n")

    for mode in ["zero", "few", "cot"]:
        summarize(text, mode=mode)
        print()

def read_file(filepath: str) -> str:
    """Read text from a .txt file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return ""

def interactive_mode():
    """Let the user type or paste text directly."""
    print("\n📝 Paste your text below.")
    print("When done, type END on a new line and press Enter.\n")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Document Summarizer")
    parser.add_argument("--file",    type=str, help="Path to a .txt file to summarize")
    parser.add_argument("--mode",    type=str, default="zero", choices=["zero", "few", "cot"], help="Prompting strategy")
    parser.add_argument("--style",   type=str, choices=["eli5", "technical", "executive"], help="Output style")
    parser.add_argument("--compare", action="store_true", help="Run all 3 strategies and compare")
    args = parser.parse_args()

    # Get the text
    if args.file:
        text = read_file(args.file)
    else:
        text = interactive_mode()

    if not text:
        print("No text provided. Exiting.")
        exit()

    # Run summarizer
    if args.compare:
        compare_all(text)
    else:
        summarize(text, mode=args.mode, style=args.style)