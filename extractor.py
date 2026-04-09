import json
import ollama
import argparse
from config import MODEL
from schemas.models import (
    ExtractionResult,
    ContactInfo,
    MeetingNotes,
    ProductReview
)
from validators import parse_and_validate, validate_with_retry


# ── PROMPT BUILDERS ────────────────────────────────────────────────────────

def build_extraction_prompt(text: str, attempt: int = 1) -> str:
    """Build a prompt that extracts named entities as structured JSON."""
    strictness = ""
    if attempt > 1:
        strictness = "\nCRITICAL: Return ONLY valid JSON. No extra text. No markdown."

    return f"""Extract structured information from the text below.
Return ONLY a valid JSON object matching this exact schema.
Do not include any explanation or text outside the JSON.{strictness}

Schema:
{{
  "persons": ["list of person names"],
  "organizations": ["list of organization names"],
  "locations": ["list of locations"],
  "dates": ["list of dates or time periods"],
  "key_facts": ["3 most important facts"],
  "sentiment": "positive OR negative OR neutral",
  "summary": "one sentence summary"
}}

Text to analyze:
{text}

JSON output:"""


def build_contact_prompt(text: str, attempt: int = 1) -> str:
    """Build a prompt for extracting contact information."""
    return f"""Extract contact information from the text below.
Return ONLY a valid JSON object. No extra text.

Schema:
{{
  "name": "full name or null",
  "email": "email address or null",
  "phone": "phone number or null",
  "company": "company name or null",
  "role": "job title or null",
  "location": "city or country or null"
}}

Text:
{text}

JSON output:"""


def build_meeting_prompt(text: str, attempt: int = 1) -> str:
    """Build a prompt for extracting meeting notes structure."""
    return f"""Extract structured data from these meeting notes.
Return ONLY a valid JSON object.

Schema:
{{
  "title": "meeting title or topic",
  "date": "date if mentioned or null",
  "attendees": ["list of names present"],
  "decisions": ["list of decisions made"],
  "action_items": ["list of tasks with owner if mentioned"],
  "next_meeting": "next meeting info or null"
}}

Meeting notes:
{text}

JSON output:"""


def build_review_prompt(text: str, attempt: int = 1) -> str:
    """Build a prompt for extracting product review structure."""
    return f"""Extract structured data from this product review.
Return ONLY a valid JSON object.

Schema:
{{
  "product_name": "name of product",
  "rating": number from 1 to 5 or null,
  "pros": ["positive aspects"],
  "cons": ["negative aspects"],
  "sentiment": "positive OR negative OR neutral",
  "would_recommend": true or false or null
}}

Review:
{text}

JSON output:"""


# ── EXTRACTOR FUNCTIONS ────────────────────────────────────────────────────

def call_model(prompt: str) -> str:
    """Send a prompt to Ollama and return raw response text."""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={"temperature": 0.1}  # low temp for consistent structured output
    )
    return response["message"]["content"]


def extract_entities(text: str) -> ExtractionResult:
    """Extract named entities and key facts from any text."""
    print(f"\nExtracting entities from text ({len(text)} chars)...")

    def generate(attempt):
        prompt = build_extraction_prompt(text, attempt)
        return call_model(prompt)

    result, attempts = validate_with_retry(generate, ExtractionResult)

    if result:
        print("\n✅ Extraction successful!\n")
        print(f"  Persons:       {result.persons}")
        print(f"  Organizations: {result.organizations}")
        print(f"  Locations:     {result.locations}")
        print(f"  Dates:         {result.dates}")
        print(f"  Sentiment:     {result.sentiment}")
        print(f"  Summary:       {result.summary}")
        print(f"  Key facts:")
        for fact in result.key_facts:
            print(f"    - {fact}")

    return result


def extract_contact(text: str) -> ContactInfo:
    """Extract contact information from text."""
    print(f"\nExtracting contact info...")

    raw = call_model(build_contact_prompt(text))
    result, error = parse_and_validate(raw, ContactInfo)

    if result:
        print("\n✅ Contact extracted!")
        data = result.model_dump()
        for k, v in data.items():
            if v:
                print(f"  {k.capitalize()}: {v}")
    else:
        print(f"❌ Extraction failed: {error}")

    return result


def extract_meeting(text: str) -> MeetingNotes:
    """Extract structured data from meeting notes."""
    print(f"\nExtracting meeting notes...")

    raw = call_model(build_meeting_prompt(text))
    result, error = parse_and_validate(raw, MeetingNotes)

    if result:
        print("\n✅ Meeting notes extracted!")
        print(f"  Title:       {result.title}")
        print(f"  Date:        {result.date}")
        print(f"  Attendees:   {result.attendees}")
        print(f"  Decisions:   {result.decisions}")
        print(f"  Action items:{result.action_items}")
        print(f"  Next meeting:{result.next_meeting}")
    else:
        print(f"❌ Extraction failed: {error}")

    return result


def extract_review(text: str) -> ProductReview:
    """Extract structured data from a product review."""
    print(f"\nExtracting product review...")

    raw = call_model(build_review_prompt(text))
    result, error = parse_and_validate(raw, ProductReview)

    if result:
        print("\n✅ Review extracted!")
        print(f"  Product:   {result.product_name}")
        print(f"  Rating:    {result.rating}/5")
        print(f"  Sentiment: {result.sentiment}")
        print(f"  Pros:      {result.pros}")
        print(f"  Cons:      {result.cons}")
        print(f"  Recommend: {result.would_recommend}")
    else:
        print(f"❌ Extraction failed: {error}")

    return result


def batch_extract(texts: list) -> list:
    """
    Extract entities from a list of texts.
    Returns a list of ExtractionResult objects.
    """
    results = []
    print(f"\nBatch extracting {len(texts)} texts...\n")
    for i, text in enumerate(texts, 1):
        print(f"[{i}/{len(texts)}] Processing: {text[:60]}...")
        result = extract_entities(text)
        results.append(result)
    return results


# ── SAMPLE TEXTS ──────────────────────────────────────────────────────────

SAMPLE_NEWS = """
Apple CEO Tim Cook announced on Tuesday that the company will open a new
research center in Bangalore, India by March 2025. The facility will employ
over 3,000 engineers and focus on artificial intelligence development.
Indian Prime Minister Narendra Modi welcomed the investment, calling it
a milestone for India's growing tech sector.
"""

SAMPLE_CONTACT = """
Hi, I'm Sarah Johnson, Senior Product Manager at TechCorp Solutions.
You can reach me at sarah.johnson@techcorp.com or call +1-555-0192.
I'm based in San Francisco, California.
"""

SAMPLE_MEETING = """
Q3 Planning Meeting — October 15, 2024
Attendees: Roshini, Priya, Arjun, Dev

We decided to launch the new dashboard feature by November 1st.
Arjun will complete the backend API by October 22nd.
Priya will handle UI design and deliver mockups by October 18th.
Roshini will coordinate with the QA team starting October 25th.
Next meeting scheduled for October 22nd at 10am.
"""

SAMPLE_REVIEW = """
I bought the Sony WH-1000XM5 headphones last month and I'm blown away.
The noise cancellation is absolutely incredible — best I've ever used.
Sound quality is rich and detailed. Battery lasts forever.
The only downside is they feel a bit tight after several hours of wear.
Also a little pricey at Rs. 29,000 but honestly worth every rupee.
I'd give them 5 out of 5 and definitely recommend them to anyone serious
about audio quality.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured Data Extractor")
    parser.add_argument("--entities", action="store_true", help="Extract entities from sample news")
    parser.add_argument("--contact",  action="store_true", help="Extract contact info from sample")
    parser.add_argument("--meeting",  action="store_true", help="Extract meeting notes from sample")
    parser.add_argument("--review",   action="store_true", help="Extract product review from sample")
    parser.add_argument("--all",      action="store_true", help="Run all extractors on all samples")
    parser.add_argument("--text",     type=str, help="Extract entities from custom text")
    args = parser.parse_args()

    if args.text:
        extract_entities(args.text)
    elif args.contact:
        extract_contact(SAMPLE_CONTACT)
    elif args.meeting:
        extract_meeting(SAMPLE_MEETING)
    elif args.review:
        extract_review(SAMPLE_REVIEW)
    elif args.all:
        print("\n" + "="*55)
        print("  RUNNING ALL EXTRACTORS")
        print("="*55)
        extract_entities(SAMPLE_NEWS)
        extract_contact(SAMPLE_CONTACT)
        extract_meeting(SAMPLE_MEETING)
        extract_review(SAMPLE_REVIEW)
    else:
        extract_entities(SAMPLE_NEWS)