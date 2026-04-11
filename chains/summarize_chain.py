from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def get_llm(model: str = "llama3.2", temperature: float = 0.7):
    return OllamaLLM(model=model, temperature=temperature)


def simple_summary_chain(text: str) -> str:
    llm = get_llm()
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""Summarize the following text in exactly 3 sentences.
Focus on the most important ideas.

Text: {text}

Summary:"""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


def sentiment_chain(text: str) -> str:
    llm = get_llm(temperature=0.1)
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""Analyze the sentiment of the following text.
Return ONLY one of: POSITIVE, NEGATIVE, NEUTRAL
Then give one sentence explaining why.

Text: {text}

Sentiment:"""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


def keyword_chain(text: str) -> str:
    llm = get_llm(temperature=0.1)
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""Extract exactly 5 keywords from the text below.
Return them as a comma-separated list. No explanations.

Text: {text}

Keywords:"""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


def sequential_analysis_chain(text: str) -> dict:
    print("Step 1: Summarizing...")
    summary = simple_summary_chain(text)

    print("Step 2: Analyzing sentiment of summary...")
    sentiment = sentiment_chain(summary)

    print("Step 3: Extracting keywords from summary...")
    keywords = keyword_chain(summary)

    return {
        "original_length": len(text),
        "summary": summary,
        "sentiment": sentiment,
        "keywords": keywords
    }


def map_reduce_summary(text: str) -> str:
    llm = get_llm()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks...")

    summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")
        summary = simple_summary_chain(chunk)
        summaries.append(summary)

    combined = " ".join(summaries)
    print("Combining summaries...")
    final = simple_summary_chain(combined)
    return final


if __name__ == "__main__":
    sample = """
    Artificial intelligence is transforming every industry. Companies worldwide
    are investing billions in AI research and deployment. Machine learning models
    can now perform tasks that were previously thought to require human intelligence,
    from medical diagnosis to creative writing. However, concerns about job
    displacement, bias, and safety remain significant challenges. Experts disagree
    about the timeline and impact of artificial general intelligence. Some believe
    transformative AI is decades away while others think it could arrive within years.
    Regulation is struggling to keep pace with technological development.
    """

    print("\n" + "="*55)
    print("  SEQUENTIAL ANALYSIS CHAIN")
    print("="*55)
    result = sequential_analysis_chain(sample)
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nSentiment:\n{result['sentiment']}")
    print(f"\nKeywords:\n{result['keywords']}")