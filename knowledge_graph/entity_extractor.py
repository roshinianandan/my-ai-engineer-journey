import json
import re
import ollama
from config import MODEL


def extract_entities_and_relations(text: str) -> dict:
    """
    Use LLM to extract entities and relationships from text.

    Returns a dict with:
    - entities: list of {name, type, description}
    - relations: list of {source, relation, target}
    """
    prompt = f"""Extract entities and relationships from the text below.

Text:
{text}

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "entity name", "type": "CONCEPT|TECHNOLOGY|PERSON|ORGANIZATION|PROCESS", "description": "brief description"}}
  ],
  "relations": [
    {{"source": "entity1 name", "relation": "relationship type", "target": "entity2 name"}}
  ]
}}

Rules:
- Extract 3-8 important entities
- Extract 3-8 meaningful relationships
- Relation should be a short phrase like "is part of", "uses", "created by", "enables"
- Only use entity names that appear in your entities list
- Return ONLY the JSON object, no other text"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.1}
        )
        raw = response["message"]["content"].strip()

        # Clean JSON
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        result = json.loads(raw)
        return result

    except Exception as e:
        print(f"[EntityExtractor] Extraction failed: {e}")
        return {"entities": [], "relations": []}


def extract_from_documents(documents: list) -> dict:
    """
    Extract entities and relations from multiple documents.
    Merges results into a single unified graph structure.
    """
    all_entities = {}
    all_relations = []

    for i, doc in enumerate(documents):
        print(f"[EntityExtractor] Processing document {i+1}/{len(documents)}: "
              f"{doc.get('source', 'unknown')}...")

        text = doc.get("text", "")
        if not text.strip():
            continue

        result = extract_entities_and_relations(text)

        # Merge entities (avoid duplicates by name)
        for entity in result.get("entities", []):
            name = entity["name"].lower().strip()
            if name not in all_entities:
                all_entities[name] = {
                    "name": entity["name"],
                    "type": entity.get("type", "CONCEPT"),
                    "description": entity.get("description", ""),
                    "sources": [doc.get("source", "unknown")]
                }
            else:
                # Add source if not already listed
                src = doc.get("source", "unknown")
                if src not in all_entities[name]["sources"]:
                    all_entities[name]["sources"].append(src)

        # Add relations
        for rel in result.get("relations", []):
            relation = {
                "source": rel["source"],
                "relation": rel["relation"],
                "target": rel["target"],
                "doc_source": doc.get("source", "unknown")
            }
            # Avoid exact duplicates
            if relation not in all_relations:
                all_relations.append(relation)

    print(f"[EntityExtractor] Extracted {len(all_entities)} entities, "
          f"{len(all_relations)} relations")

    return {
        "entities": list(all_entities.values()),
        "relations": all_relations
    }