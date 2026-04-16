import ollama
import os
from pathlib import Path
from multimodal.image_analyzer import encode_image_to_base64

VISION_MODEL = "llava"


def extract_text_from_image(image_path: str) -> dict:
    """
    OCR using LLaVA — extract all text visible in an image.
    Works on screenshots, photos of documents, signs, etc.
    """
    if not Path(image_path).exists():
        return {"error": f"Image not found: {image_path}", "success": False}

    image_b64 = encode_image_to_base64(image_path)

    print(f"Extracting text from: {image_path}")

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": """Extract ALL text visible in this image.
Return the text exactly as it appears, preserving formatting where possible.
If there is no text, say 'No text found'.
Do not add any commentary — only return the extracted text.""",
            "images": [image_b64]
        }]
    )

    extracted_text = response["message"]["content"]
    print(f"Extracted: {extracted_text[:200]}...")

    return {
        "image_path": image_path,
        "extracted_text": extracted_text,
        "char_count": len(extracted_text),
        "success": True
    }


def extract_structured_data_from_image(
    image_path: str,
    data_type: str = "table"
) -> dict:
    """
    Extract structured data from an image.
    data_type: 'table', 'form', 'receipt', 'business_card'
    """
    if not Path(image_path).exists():
        return {"error": f"Image not found: {image_path}"}

    image_b64 = encode_image_to_base64(image_path)

    prompts = {
        "table": "Extract all data from the table in this image. Return it as a structured list with headers and rows.",
        "form": "Extract all form fields and their values from this image. Return as field: value pairs.",
        "receipt": "Extract all information from this receipt: store name, items, prices, total, date, and any other relevant details.",
        "business_card": "Extract all information from this business card: name, title, company, email, phone, address, and website."
    }

    prompt = prompts.get(data_type, prompts["table"])

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_b64]
        }]
    )

    return {
        "image_path": image_path,
        "data_type": data_type,
        "extracted_data": response["message"]["content"],
        "success": True
    }


def image_to_rag_document(image_path: str) -> dict:
    """
    Full pipeline:
    1. Extract text from image using OCR
    2. Pass extracted text to the RAG pipeline
    3. Return the combined result

    This enables querying documents that only exist as images.
    """
    print(f"\nProcessing image for RAG: {image_path}")

    ocr_result = extract_text_from_image(image_path)

    if not ocr_result["success"] or not ocr_result["extracted_text"]:
        return {"error": "Could not extract text from image", "success": False}

    extracted_text = ocr_result["extracted_text"]

    if extracted_text.strip() == "No text found":
        return {
            "image_path": image_path,
            "message": "No text found in image",
            "success": False
        }

    print(f"Extracted {len(extracted_text)} characters of text")
    print("Text ready to be added to RAG pipeline")

    return {
        "image_path": image_path,
        "extracted_text": extracted_text,
        "char_count": len(extracted_text),
        "success": True,
        "next_step": "Pass extracted_text to rag.knowledge_base.ingest_text()"
    }


def batch_ocr(folder: str, output_file: str = None) -> list:
    """
    Run OCR on all images in a folder and collect results.
    Optionally save all extracted text to a single file.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"Folder not found: {folder}")
        return []

    extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    images = [f for f in folder_path.iterdir()
              if f.suffix.lower() in extensions]

    print(f"Running batch OCR on {len(images)} images...")
    results = []

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {image_path.name}")
        result = extract_text_from_image(str(image_path))
        results.append(result)

    if output_file and results:
        with open(output_file, "w") as f:
            for r in results:
                f.write(f"=== {r.get('image_path', 'unknown')} ===\n")
                f.write(r.get("extracted_text", "") + "\n\n")
        print(f"\nAll text saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OCR Pipeline using LLaVA")
    parser.add_argument("--image",     type=str, help="Extract text from image")
    parser.add_argument("--structured",type=str, help="Extract structured data from image")
    parser.add_argument("--type",      type=str, default="table",
                        choices=["table", "form", "receipt", "business_card"])
    parser.add_argument("--to-rag",    type=str, help="Extract text and prepare for RAG")
    parser.add_argument("--batch",     type=str, help="Run OCR on all images in folder")
    parser.add_argument("--output",    type=str, help="Save batch OCR results to file")
    args = parser.parse_args()

    if args.batch:
        results = batch_ocr(args.batch, output_file=args.output)
        print(f"\nProcessed {len(results)} images")
    elif args.to_rag:
        result = image_to_rag_document(args.to_rag)
        print(f"\nResult: {result}")
    elif args.structured:
        result = extract_structured_data_from_image(args.structured, args.type)
        print(f"\nExtracted:\n{result['extracted_data']}")
    elif args.image:
        result = extract_text_from_image(args.image)
        print(f"\nExtracted text:\n{result['extracted_text']}")
    else:
        print("Usage:")
        print("  python multimodal/ocr_pipeline.py --image path/to/image.jpg")
        print("  python multimodal/ocr_pipeline.py --structured receipt.jpg --type receipt")
        print("  python multimodal/ocr_pipeline.py --to-rag document_photo.jpg")
        print("  python multimodal/ocr_pipeline.py --batch data/images/ --output ocr_results.txt")