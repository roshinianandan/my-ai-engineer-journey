import json
import random
import ollama
from config import MODEL

# ── RAW DATA ─────────────────────────────────────────────────────────────
# These are seed examples in our target domain — AIML concepts.
# In a real project you would have hundreds of domain-specific seeds.
# Quality of seeds directly determines quality of fine-tuned model.

SEED_QA_PAIRS = [
    {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve their performance on tasks without being explicitly programmed. It uses statistical techniques to identify patterns and make decisions."
    },
    {
        "question": "Explain the difference between supervised and unsupervised learning.",
        "answer": "Supervised learning trains models on labeled data where the correct output is known, such as classifying emails as spam or not spam. Unsupervised learning works with unlabeled data to discover hidden patterns or structure, such as customer segmentation."
    },
    {
        "question": "What is a neural network?",
        "answer": "A neural network is a computational model inspired by the structure of biological brains. It consists of layers of interconnected nodes called neurons that process information. Each connection has a weight that is adjusted during training to minimize prediction errors."
    },
    {
        "question": "What is RAG in AI?",
        "answer": "RAG stands for Retrieval Augmented Generation. It is a technique that combines a retrieval system with a language model. The retrieval system finds relevant documents from a knowledge base and passes them as context to the LLM, which generates a grounded answer based on the retrieved information."
    },
    {
        "question": "What are embeddings?",
        "answer": "Embeddings are dense numerical vector representations of text that capture semantic meaning. Similar texts produce similar vectors. They are created by neural networks trained to place semantically related content close together in a high-dimensional vector space."
    },
    {
        "question": "What is the transformer architecture?",
        "answer": "The transformer architecture is a neural network design that uses attention mechanisms to process sequences in parallel rather than sequentially. It was introduced in the paper Attention Is All You Need and became the foundation of modern language models like GPT and BERT."
    },
    {
        "question": "What is fine-tuning in machine learning?",
        "answer": "Fine-tuning is the process of taking a pretrained model and continuing its training on a smaller, task-specific dataset. It transfers the general knowledge from pretraining while specializing the model for a particular domain or task. It requires far less data and compute than training from scratch."
    },
    {
        "question": "What is overfitting and how do you prevent it?",
        "answer": "Overfitting occurs when a model learns the training data too well, including its noise and random fluctuations, leading to poor performance on new unseen data. Prevention techniques include dropout, regularization, early stopping, data augmentation, and using more training data."
    },
    {
        "question": "What is a vector database?",
        "answer": "A vector database is a specialized database system designed to store, manage, and search high-dimensional embedding vectors. It supports fast similarity search using algorithms like HNSW, making it essential for RAG systems, semantic search, and recommendation engines."
    },
    {
        "question": "Explain gradient descent.",
        "answer": "Gradient descent is an optimization algorithm used to train neural networks. It calculates the gradient of the loss function with respect to each model parameter and updates the parameters in the direction that minimizes the loss. The learning rate controls how large each update step is."
    },
    {
        "question": "What is prompt engineering?",
        "answer": "Prompt engineering is the practice of designing and optimizing input instructions to get the best possible output from a language model. Techniques include zero-shot prompting, few-shot prompting with examples, chain-of-thought reasoning, and system prompt design."
    },
    {
        "question": "What is the difference between a parameter and a hyperparameter?",
        "answer": "Parameters are the values that a model learns during training, such as neural network weights and biases. Hyperparameters are settings that are configured before training begins, such as learning rate, batch size, and number of layers. Parameters are optimized automatically while hyperparameters require manual tuning."
    },
    {
        "question": "What is tokenization?",
        "answer": "Tokenization is the process of breaking text into smaller units called tokens before processing by a language model. Tokens can be words, subwords, or characters depending on the tokenizer. Each token is mapped to a unique integer ID from the model's vocabulary."
    },
    {
        "question": "What is attention mechanism?",
        "answer": "The attention mechanism allows neural networks to focus on different parts of the input when producing each part of the output. Self-attention enables each token to attend to all other tokens in the sequence, capturing long-range dependencies. It is the core innovation in transformer models."
    },
    {
        "question": "What is transfer learning?",
        "answer": "Transfer learning is a technique where knowledge gained from training a model on one task is applied to a different but related task. Instead of training from scratch, you start with a pretrained model and adapt it. This dramatically reduces the data and compute needed for the new task."
    },
]


def generate_variations(qa_pair: dict, num_variations: int = 3) -> list:
    """
    Use the LLM to generate variations of an existing Q&A pair.
    This is called data augmentation — expanding a small dataset
    by creating semantically equivalent but differently phrased examples.
    """
    prompt = f"""Given this question and answer pair, generate {num_variations} different ways
to ask the same question with slightly different phrasing.
For each variation also write a slightly different but equally correct answer.
Keep the same factual content but vary the wording and style.

Original Question: {qa_pair['question']}
Original Answer: {qa_pair['answer']}

Return ONLY a JSON array with {num_variations} objects, each with 'question' and 'answer' keys.
No extra text, just the JSON array."""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.8}
        )
        raw = response["message"]["content"]

        # Clean and parse JSON
        raw = raw.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        variations = json.loads(raw)
        return variations if isinstance(variations, list) else []

    except Exception as e:
        print(f"  [Variation generation failed: {e}]")
        return []


def generate_new_examples(topic: str, num_examples: int = 5) -> list:
    """
    Generate completely new Q&A pairs on a specific topic.
    Used to expand the dataset into areas not covered by seeds.
    """
    prompt = f"""Generate {num_examples} high-quality question and answer pairs about: {topic}

Requirements:
- Questions should be what a student learning AIML would ask
- Answers should be accurate, clear, and 2-4 sentences long
- Cover different aspects of the topic
- Vary the difficulty from beginner to intermediate

Return ONLY a JSON array. Each item must have 'question' and 'answer' keys.
No extra text outside the JSON array."""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.7}
        )
        raw = response["message"]["content"].strip()

        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        examples = json.loads(raw)
        return examples if isinstance(examples, list) else []

    except Exception as e:
        print(f"  [New example generation failed: {e}]")
        return []


def build_raw_dataset(
    use_seeds: bool = True,
    augment: bool = True,
    generate_new: bool = True,
    target_size: int = 100
) -> list:
    """
    Build a raw dataset of Q&A pairs.

    1. Start with seed examples
    2. Augment with variations of each seed
    3. Generate new examples on related topics
    4. Shuffle and return
    """
    dataset = []

    print(f"\n{'='*55}")
    print("  BUILDING RAW DATASET")
    print(f"{'='*55}\n")

    # Step 1: Add seeds
    if use_seeds:
        print(f"Step 1: Adding {len(SEED_QA_PAIRS)} seed examples...")
        dataset.extend(SEED_QA_PAIRS)

    # Step 2: Augment each seed with variations
    if augment and len(dataset) < target_size:
        print(f"\nStep 2: Generating variations for each seed...")
        variations_needed = min(
            len(SEED_QA_PAIRS),
            (target_size - len(dataset)) // 3
        )
        for i, pair in enumerate(SEED_QA_PAIRS[:variations_needed]):
            print(f"  Generating variations for: {pair['question'][:50]}...")
            variations = generate_variations(pair, num_variations=2)
            dataset.extend(variations)
            print(f"  Added {len(variations)} variations. Total: {len(dataset)}")

    # Step 3: Generate new examples on related topics
    if generate_new and len(dataset) < target_size:
        topics = [
            "deep learning architectures",
            "natural language processing basics",
            "model evaluation metrics",
            "data preprocessing techniques",
            "reinforcement learning concepts"
        ]
        print(f"\nStep 3: Generating new examples on {len(topics)} topics...")
        for topic in topics:
            if len(dataset) >= target_size:
                break
            print(f"  Generating examples for: {topic}...")
            new_examples = generate_new_examples(topic, num_examples=5)
            dataset.extend(new_examples)
            print(f"  Added {len(new_examples)} examples. Total: {len(dataset)}")

    # Shuffle for good measure
    random.shuffle(dataset)

    print(f"\nRaw dataset size: {len(dataset)} examples")
    return dataset


def validate_example(example: dict) -> tuple:
    """
    Validate a single Q&A example.
    Returns (is_valid, reason)
    """
    if not isinstance(example, dict):
        return False, "Not a dict"
    if "question" not in example or "answer" not in example:
        return False, "Missing question or answer field"
    if len(example["question"].strip()) < 10:
        return False, "Question too short"
    if len(example["answer"].strip()) < 20:
        return False, "Answer too short"
    if len(example["question"]) > 500:
        return False, "Question too long"
    if len(example["answer"]) > 2000:
        return False, "Answer too long"

    return True, "OK"


def clean_dataset(dataset: list) -> list:
    """
    Remove invalid, duplicate, or low-quality examples.
    """
    cleaned = []
    seen_questions = set()
    removed = 0

    for example in dataset:
        is_valid, reason = validate_example(example)

        if not is_valid:
            removed += 1
            continue

        # Deduplicate by question text
        q_normalized = example["question"].lower().strip()
        if q_normalized in seen_questions:
            removed += 1
            continue

        seen_questions.add(q_normalized)
        cleaned.append({
            "question": example["question"].strip(),
            "answer": example["answer"].strip()
        })

    print(f"\nDataset cleaning: {len(dataset)} → {len(cleaned)} examples ({removed} removed)")
    return cleaned