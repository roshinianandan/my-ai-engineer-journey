from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chains.summarize_chain import (
    simple_summary_chain,
    sentiment_chain,
    keyword_chain,
    sequential_analysis_chain
)


def get_llm(temperature: float = 0.5):
    return OllamaLLM(model="llama3.2", temperature=temperature)


def translation_chain(text: str, target_language: str = "Tamil") -> str:
    llm = get_llm(temperature=0.3)
    prompt = PromptTemplate(
        input_variables=["text", "language"],
        template="Translate the following text to {language}. Return only the translation.\n\nText: {text}\n\nTranslation:"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text, "language": target_language})


def quiz_chain(text: str, num_questions: int = 3) -> str:
    llm = get_llm(temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["text", "num"],
        template="""Generate {num} multiple choice quiz questions based on this text.
For each question provide 4 options labeled A, B, C, D and mark the correct answer.

Text: {text}

Quiz:"""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text, "num": str(num_questions)})


def full_document_pipeline(text: str, translate_to: str = None) -> dict:
    print("\n" + "="*60)
    print("  FULL DOCUMENT PIPELINE")
    print("="*60)

    print("\n[1/4] Generating summary...")
    summary = simple_summary_chain(text)

    print("[2/4] Analyzing sentiment...")
    sentiment = sentiment_chain(summary)

    print("[3/4] Extracting keywords...")
    keywords = keyword_chain(summary)

    print("[4/4] Generating quiz questions...")
    quiz = quiz_chain(text, num_questions=3)

    result = {
        "original_length": len(text),
        "summary": summary,
        "sentiment": sentiment,
        "keywords": keywords,
        "quiz": quiz
    }

    if translate_to:
        print(f"[5/5] Translating summary to {translate_to}...")
        result["translation"] = translation_chain(summary, translate_to)
        result["translated_to"] = translate_to

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)

    print(f"\nSummary:\n{summary}")
    print(f"\nSentiment: {sentiment}")
    print(f"\nKeywords: {keywords}")
    print(f"\nQuiz:\n{quiz}")
    if translate_to and "translation" in result:
        print(f"\nTranslation ({translate_to}):\n{result['translation']}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LangChain Document Pipeline")
    parser.add_argument("--text",      type=str, help="Text to process")
    parser.add_argument("--file",      type=str, help="Text file to process")
    parser.add_argument("--translate", type=str, help="Target language")
    parser.add_argument("--rag",       action="store_true", help="Run RAG chain")
    args = parser.parse_args()

    if args.rag:
        from chains.qa_chain import build_rag_chain, run_qa
        print("Building LangChain RAG...")
        chain = build_rag_chain()
        if chain:
            while True:
                q = input("\nQuestion (or quit): ").strip()
                if q.lower() == "quit":
                    break
                run_qa(q, chain)
    else:
        if args.file:
            with open(args.file, "r") as f:
                text = f.read()
        elif args.text:
            text = args.text
        else:
            text = """Machine learning is transforming how we interact with technology.
            Modern AI systems can understand language, generate images, and solve complex problems.
            The field advances rapidly with new models and techniques emerging constantly.
            However ethical considerations around bias, privacy and safety must be addressed."""

        full_document_pipeline(text, translate_to=args.translate)