# Zero-shot prompt — no examples, just clear instructions
ZERO_SHOT_SUMMARY = """
You are an expert document analyst.

Analyze the following text and return your response in EXACTLY this format:

SUMMARY:
Write a clear 3-sentence summary here.

KEY POINTS:
- Point 1
- Point 2
- Point 3
- Point 4
- Point 5

SENTIMENT:
One word only: Positive / Negative / Neutral

TOPIC:
One short phrase describing the main topic.

Text to analyze:
{text}
"""

# Few-shot prompt — teaches the model with examples first
FEW_SHOT_SUMMARY = """
You are a document summarizer. Here are examples of how to summarize:

EXAMPLE 1:
Input: "The stock market crashed today losing 500 points. Investors panicked and sold holdings rapidly. Analysts predict further decline next week."
Output:
SUMMARY: Markets fell sharply today as panic selling gripped investors. A 500-point drop triggered widespread portfolio liquidation. Experts warn conditions may worsen in coming days.
SENTIMENT: Negative

EXAMPLE 2:
Input: "Scientists discovered a new species of bird in the Amazon rainforest. The vibrant blue creature was spotted by a research team. It represents the first new bird discovery in the region in 15 years."
Output:
SUMMARY: Researchers uncovered a previously unknown bird species in the Amazon. The colorful blue bird marks the first such discovery in 15 years. The finding has excited the global ornithology community.
SENTIMENT: Positive

Now do the same for this text:
Input: "{text}"
Output:
"""

# Chain-of-thought prompt — forces step-by-step reasoning
CHAIN_OF_THOUGHT_SUMMARY = """
You are an expert analyst. Think through this carefully step by step before giving your final answer.

Step 1: Read the full text and identify what it is about.
Step 2: Find the 3 most important ideas.
Step 3: Determine the emotional tone.
Step 4: Write a clean summary using your analysis.

Text:
{text}

Now work through each step and then give your final structured output:

REASONING:
(your step by step thinking here)

FINAL SUMMARY:
(3 sentences)

KEY POINTS:
- Point 1
- Point 2
- Point 3

SENTIMENT: (Positive / Negative / Neutral)
"""

# Style-based prompts
STYLE_PROMPTS = {
    "eli5": """
Explain this text like I am 5 years old. Use very simple words,
short sentences, and a fun analogy if possible.

Text: {text}

Simple explanation:
""",
    "technical": """
You are a technical writer. Summarize this text with precise,
professional language. Include any technical terms. Be concise and exact.

Text: {text}

Technical summary:
""",
    "executive": """
You are writing for a busy CEO who has 30 seconds to read this.
Give a 2-sentence summary with the single most important takeaway
and what action (if any) is needed.

Text: {text}

Executive brief:
"""
}