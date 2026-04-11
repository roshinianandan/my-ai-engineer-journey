from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


def get_llm(temperature: float = 0.3):
    return OllamaLLM(model="llama3.2", temperature=temperature)


def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


def build_rag_chain(docs_folder: str = "./data/docs"):
    import os
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    embeddings = get_embeddings()
    all_docs = []

    if os.path.exists(docs_folder):
        for filename in os.listdir(docs_folder):
            filepath = os.path.join(docs_folder, filename)
            if filename.endswith(".txt"):
                with open(filepath, "r") as f:
                    content = f.read()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = splitter.split_text(content)
                for chunk in chunks:
                    all_docs.append(Document(
                        page_content=chunk,
                        metadata={"source": filename}
                    ))

    if not all_docs:
        print("No documents found in data/docs/")
        return None

    print(f"Loaded {len(all_docs)} chunks")

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory="./chroma_langchain"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = """Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = get_llm()

    def rag_chain(question: str) -> dict:
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        sources = list(set(d.metadata.get("source", "unknown") for d in docs))
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return {"answer": answer, "sources": sources}

    return rag_chain


def run_qa(question: str, rag_chain) -> dict:
    print(f"\nQuestion: {question}")
    print("-" * 50)
    result = rag_chain(question)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    return result


if __name__ == "__main__":
    print("Building RAG chain with LangChain...")
    chain = build_rag_chain()

    if chain:
        questions = [
            "What is machine learning?",
            "What is RAG and how does it work?",
            "What Python libraries are used for data science?"
        ]
        for q in questions:
            run_qa(q, chain)
            print()