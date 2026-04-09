import json
import re
from pydantic import BaseModel, ValidationError


def clean_json_string(raw: str) -> str:
    """
    Clean common LLM JSON output issues before parsing.
    LLMs often wrap JSON in markdown code blocks or add extra text.
    """
    # Remove markdown code blocks
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)

    # Find the first { and last } to extract just the JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    return raw.strip()


def parse_and_validate(raw_output: str, model_class: type) -> tuple:
    """
    Parse LLM output as JSON and validate against a Pydantic model.

    Returns: (parsed_object, error_message)
    If successful: (ExtractionResult(...), None)
    If failed:     (None, "error description")
    """
    try:
        cleaned = clean_json_string(raw_output)
        data = json.loads(cleaned)
        obj = model_class(**data)
        return obj, None

    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

    except ValidationError as e:
        return None, f"Schema validation error: {e}"

    except Exception as e:
        return None, f"Unexpected error: {e}"


def validate_with_retry(
    generate_fn,
    model_class: type,
    max_retries: int = 3
) -> tuple:
    """
    Call generate_fn() and validate the output.
    If validation fails, retry up to max_retries times with a clearer prompt.

    generate_fn: a callable that returns a raw LLM string
    Returns: (validated_object, attempts_taken)
    """
    for attempt in range(1, max_retries + 1):
        raw = generate_fn(attempt)
        obj, error = parse_and_validate(raw, model_class)

        if obj is not None:
            print(f"Validation passed on attempt {attempt}")
            return obj, attempt

        print(f"Attempt {attempt} failed: {error}")
        if attempt < max_retries:
            print("Retrying with stronger instruction...\n")

    print(f"Failed after {max_retries} attempts.")
    return None, max_retries