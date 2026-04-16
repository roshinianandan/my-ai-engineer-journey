import base64
import ollama
import argparse
from pathlib import Path
from PIL import Image
import io


VISION_MODEL = "llava"


def encode_image_to_base64(image_path: str) -> str:
    """
    Convert an image file to base64 string.
    This is how images are sent to LLMs — converted to text
    so they can be included in the API request body.
    """
    with open(image_path, "rb") as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode("utf-8")


def get_image_info(image_path: str) -> dict:
    """Get basic metadata about an image using PIL."""
    try:
        img = Image.open(image_path)
        return {
            "path": image_path,
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "width": img.size[0],
            "height": img.size[1]
        }
    except Exception as e:
        return {"path": image_path, "error": str(e)}


def analyze_image(image_path: str, question: str = None) -> dict:
    """
    Send an image to LLaVA and get a description or answer.
    If no question is provided, generates a general description.
    """
    if not Path(image_path).exists():
        return {"error": f"Image not found: {image_path}"}

    image_b64 = encode_image_to_base64(image_path)
    image_info = get_image_info(image_path)

    prompt = question if question else (
        "Describe this image in detail. Include: "
        "what you see, colors, objects, text if any, "
        "and the overall context or setting."
    )

    print(f"\nAnalyzing: {image_path}")
    print(f"Question: {prompt[:80]}...")
    print(f"Image: {image_info.get('width')}x{image_info.get('height')} pixels")
    print("-" * 50)

    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64]
                }
            ]
        )

        answer = response["message"]["content"]
        print(f"Answer: {answer}\n")

        return {
            "image_path": image_path,
            "question": prompt,
            "answer": answer,
            "image_info": image_info,
            "success": True
        }

    except Exception as e:
        return {
            "image_path": image_path,
            "error": str(e),
            "success": False
        }


def image_qa_session(image_path: str):
    """
    Interactive Q&A session about a single image.
    Ask multiple questions about the same image.
    """
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return

    image_b64 = encode_image_to_base64(image_path)
    info = get_image_info(image_path)

    print(f"\n🖼️  Image Q&A Session")
    print(f"   Image: {image_path}")
    print(f"   Size: {info.get('width')}x{info.get('height')}")
    print("   Type 'quit' to exit\n")
    print("-" * 50)

    # First auto-describe the image
    print("Generating initial description...")
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": "Describe this image in 3 sentences.",
            "images": [image_b64]
        }]
    )
    print(f"\n🤖 Description: {response['message']['content']}\n")

    while True:
        try:
            question = input("\nAsk about this image: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue

        if question.lower() == "quit":
            break

        try:
            response = ollama.chat(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": question,
                    "images": [image_b64]
                }]
            )
            print(f"\n🤖 Answer: {response['message']['content']}")

        except Exception as e:
            print(f"Error: {e}")


def compare_images(image_path1: str, image_path2: str) -> dict:
    """
    Compare two images and describe the differences.
    """
    if not Path(image_path1).exists() or not Path(image_path2).exists():
        return {"error": "One or both images not found"}

    img1_b64 = encode_image_to_base64(image_path1)
    img2_b64 = encode_image_to_base64(image_path2)

    prompt = """I am showing you two images.
    Describe the key similarities and differences between them.
    Image 1 comes first, Image 2 comes second."""

    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [img1_b64, img2_b64]
            }]
        )

        comparison = response["message"]["content"]
        print(f"\nComparison:\n{comparison}")

        return {
            "image1": image_path1,
            "image2": image_path2,
            "comparison": comparison,
            "success": True
        }

    except Exception as e:
        return {"error": str(e), "success": False}


def batch_analyze(image_folder: str, question: str = None) -> list:
    """
    Analyze all images in a folder.
    Returns a list of analysis results.
    """
    folder = Path(image_folder)
    if not folder.exists():
        print(f"Folder not found: {image_folder}")
        return []

    extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    images = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    if not images:
        print(f"No images found in {image_folder}")
        return []

    print(f"\nBatch analyzing {len(images)} images...")
    results = []

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {image_path.name}")
        result = analyze_image(str(image_path), question)
        results.append(result)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Analyzer with LLaVA")
    parser.add_argument("--image",   type=str, help="Path to image file")
    parser.add_argument("--question",type=str, help="Question about the image")
    parser.add_argument("--qa",      action="store_true", help="Start interactive Q&A session")
    parser.add_argument("--batch",   type=str, help="Folder of images to batch analyze")
    parser.add_argument("--compare", nargs=2, metavar=("IMG1","IMG2"), help="Compare two images")
    args = parser.parse_args()

    if args.compare:
        compare_images(args.compare[0], args.compare[1])
    elif args.batch:
        results = batch_analyze(args.batch, args.question)
        print(f"\n\nAnalyzed {len(results)} images")
    elif args.qa and args.image:
        image_qa_session(args.image)
    elif args.image:
        analyze_image(args.image, args.question)
    else:
        print("Usage:")
        print("  python multimodal/image_analyzer.py --image path/to/image.jpg")
        print("  python multimodal/image_analyzer.py --image img.jpg --question 'What colors are in this image?'")
        print("  python multimodal/image_analyzer.py --image img.jpg --qa")
        print("  python multimodal/image_analyzer.py --batch data/images/")
        print("  python multimodal/image_analyzer.py --compare img1.jpg img2.jpg")