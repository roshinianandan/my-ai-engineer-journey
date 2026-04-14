import json
import os
import random
from finetuning.data_prep import (
    build_raw_dataset,
    clean_dataset,
    SEED_QA_PAIRS
)


DATA_DIR = "./finetuning/data"


def format_as_chat(example: dict) -> dict:
    """
    Format a Q&A pair into the chat completion format.
    This is the standard format for fine-tuning instruction-following models.

    The format teaches the model to behave as an AI assistant:
    - system: defines the model's role
    - user: the question
    - assistant: the expected response
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant specializing in machine learning and AI concepts. Provide clear, accurate, and educational responses."
            },
            {
                "role": "user",
                "content": example["question"]
            },
            {
                "role": "assistant",
                "content": example["answer"]
            }
        ]
    }


def format_as_completion(example: dict) -> dict:
    """
    Format as a simple prompt-completion pair.
    Used for older fine-tuning APIs or completion-style models.
    """
    return {
        "prompt": f"Question: {example['question']}\n\nAnswer:",
        "completion": f" {example['answer']}"
    }


def save_jsonl(data: list, filepath: str):
    """
    Save data as JSONL — one JSON object per line.
    JSONL is the standard format for fine-tuning datasets.
    Each line is a complete, valid JSON object.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} examples to {filepath}")


def load_jsonl(filepath: str) -> list:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def split_dataset(
    data: list,
    train_ratio: float = 0.9
) -> tuple:
    """
    Split dataset into training and validation sets.
    Always shuffle before splitting to avoid ordering bias.
    """
    shuffled = data.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]

    return train, val


def validate_jsonl_file(filepath: str) -> dict:
    """
    Validate a JSONL file before submission.
    Checks format, required fields, and token estimate.
    """
    data = load_jsonl(filepath)
    issues = []
    total_chars = 0

    for i, item in enumerate(data):
        if "messages" not in item:
            issues.append(f"Line {i+1}: Missing 'messages' field")
            continue

        messages = item["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            issues.append(f"Line {i+1}: messages must be a list with at least 2 items")
            continue

        roles = [m.get("role") for m in messages]
        if "user" not in roles or "assistant" not in roles:
            issues.append(f"Line {i+1}: Must have both user and assistant messages")

        for msg in messages:
            total_chars += len(msg.get("content", ""))

    # Rough token estimate: 1 token ≈ 4 chars
    total_tokens_est = total_chars // 4

    return {
        "total_examples": len(data),
        "issues_found": len(issues),
        "issues": issues[:5],  # show first 5 issues
        "estimated_tokens": total_tokens_est,
        "estimated_training_cost_note": "Actual cost depends on provider and model",
        "is_valid": len(issues) == 0
    }


def print_sample(filepath: str, n: int = 3):
    """Print n sample examples from a JSONL file."""
    data = load_jsonl(filepath)
    samples = random.sample(data, min(n, len(data)))

    print(f"\n{'='*55}")
    print(f"  SAMPLE EXAMPLES FROM {os.path.basename(filepath)}")
    print(f"{'='*55}\n")

    for i, item in enumerate(samples, 1):
        print(f"Example {i}:")
        for msg in item["messages"]:
            role = msg["role"].upper()
            content = msg["content"]
            if len(content) > 150:
                content = content[:150] + "..."
            print(f"  {role}: {content}")
        print()


def build_and_save_dataset(
    target_size: int = 100,
    augment: bool = True,
    generate_new: bool = True,
    format_type: str = "chat"
) -> dict:
    """
    Full pipeline: generate → clean → format → split → save → validate.
    This is the complete data preparation workflow for fine-tuning.
    """
    print(f"\n{'='*55}")
    print("  FINE-TUNING DATA PIPELINE")
    print(f"  Target size: {target_size}")
    print(f"  Format: {format_type}")
    print(f"{'='*55}")

    # Step 1: Build raw dataset
    raw = build_raw_dataset(
        augment=augment,
        generate_new=generate_new,
        target_size=target_size
    )

    # Step 2: Clean dataset
    cleaned = clean_dataset(raw)

    if len(cleaned) < 10:
        print("Warning: Very few examples. Try increasing target_size or enabling augmentation.")

    # Step 3: Format
    print(f"\nFormatting as {format_type}...")
    if format_type == "chat":
        formatted = [format_as_chat(ex) for ex in cleaned]
    else:
        formatted = [format_as_completion(ex) for ex in cleaned]

    # Step 4: Split
    train, val = split_dataset(formatted, train_ratio=0.9)
    print(f"\nSplit: {len(train)} train / {len(val)} validation")

    # Step 5: Save
    train_path = f"{DATA_DIR}/train.jsonl"
    val_path = f"{DATA_DIR}/val.jsonl"
    save_jsonl(train, train_path)
    save_jsonl(val, val_path)

    # Step 6: Validate
    print("\nValidating training file...")
    train_validation = validate_jsonl_file(train_path)
    print(f"  Valid: {train_validation['is_valid']}")
    print(f"  Examples: {train_validation['total_examples']}")
    print(f"  Estimated tokens: {train_validation['estimated_tokens']:,}")
    if train_validation["issues"]:
        print(f"  Issues: {train_validation['issues']}")

    # Step 7: Show samples
    print_sample(train_path, n=2)

    return {
        "train_file": train_path,
        "val_file": val_path,
        "train_examples": len(train),
        "val_examples": len(val),
        "total_examples": len(cleaned),
        "validation": train_validation
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-Tuning Data Pipeline")
    parser.add_argument("--build",    action="store_true", help="Build and save full dataset")
    parser.add_argument("--validate", type=str, help="Validate a JSONL file")
    parser.add_argument("--sample",   type=str, help="Print sample from a JSONL file")
    parser.add_argument("--seeds",    action="store_true", help="Only use seed examples, no generation")
    parser.add_argument("--size",     type=int, default=100, help="Target dataset size")
    args = parser.parse_args()

    if args.validate:
        result = validate_jsonl_file(args.validate)
        print(f"\nValidation result: {json.dumps(result, indent=2)}")
    elif args.sample:
        print_sample(args.sample)
    elif args.build:
        result = build_and_save_dataset(
            target_size=args.size,
            augment=not args.seeds,
            generate_new=not args.seeds
        )
        print(f"\nDataset ready: {result}")
    else:
        # Default: build seeds only for quick test
        print("Building dataset from seeds only (fast mode)...")
        import os
        os.makedirs(DATA_DIR, exist_ok=True)

        seeds_formatted = [format_as_chat(ex) for ex in SEED_QA_PAIRS]
        train, val = split_dataset(seeds_formatted)

        save_jsonl(train, f"{DATA_DIR}/train.jsonl")
        save_jsonl(val, f"{DATA_DIR}/val.jsonl")

        result = validate_jsonl_file(f"{DATA_DIR}/train.jsonl")
        print(f"\nSeed dataset ready: {len(seeds_formatted)} examples")
        print(f"Validation: {result}")