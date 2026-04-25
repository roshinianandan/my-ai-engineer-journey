import streamlit as st
import requests
import json
import time
import os

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-key-aiml-journey-2024")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

st.set_page_config(
    page_title="My AI Engineer Journey",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1A56DB, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .feature-card {
        background: #F8FAFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .status-healthy { color: #059669; font-weight: bold; }
    .status-error   { color: #DC2626; font-weight: bold; }
    .metric-value   { font-size: 1.8rem; font-weight: bold; color: #1A56DB; }
    .metric-label   { font-size: 0.85rem; color: #6B7280; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_api_health() -> dict:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def chat_api(message: str, persona: str, history: list) -> dict:
    try:
        response = requests.post(
            f"{API_URL}/chat/",
            headers=HEADERS,
            json={
                "message": message,
                "persona": persona,
                "history": history,
                "user_id": "streamlit_user"
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def rag_api(question: str, top_k: int = 3) -> dict:
    try:
        response = requests.post(
            f"{API_URL}/rag/ask",
            headers=HEADERS,
            json={
                "question": question,
                "top_k": top_k,
                "show_sources": True
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def summarize_api(text: str, style: str) -> dict:
    try:
        response = requests.post(
            f"{API_URL}/rag/summarize",
            headers=HEADERS,
            json={"text": text, "style": style},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def extract_api(text: str, extract_type: str) -> dict:
    try:
        response = requests.post(
            f"{API_URL}/rag/extract",
            headers=HEADERS,
            json={"text": text, "extract_type": extract_type},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def get_documents() -> dict:
    try:
        response = requests.get(
            f"{API_URL}/rag/documents",
            headers=HEADERS,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {"documents": [], "count": 0}
    except Exception:
        return {"documents": [], "count": 0}


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 My AI Engineer Journey")
    st.markdown("*30 days of Gen AI engineering*")
    st.divider()

    # Health status
    health = check_api_health()
    status = health.get("status", "unknown")
    color = "status-healthy" if status == "healthy" else "status-error"
    st.markdown(
        f"**API Status:** <span class='{color}'>{status.upper()}</span>",
        unsafe_allow_html=True
    )

    if status == "healthy":
        st.markdown(f"**Model:** `{health.get('model', 'unknown')}`")
        features = health.get("features", [])
        if features:
            st.markdown(f"**Features:** {', '.join(features[:3])}")

    st.divider()

    # Navigation
    page = st.selectbox(
        "Navigate",
        ["🏠 Home", "💬 Chat", "📚 RAG Q&A",
         "📝 Summarizer", "🔍 Data Extractor", "📊 Dashboard"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Quick Stats**")
    docs = get_documents()
    st.metric("Documents Indexed", docs.get("count", 0))

    st.divider()
    st.markdown("**API Endpoints**")
    st.code(f"{API_URL}", language=None)


# ── Pages ──────────────────────────────────────────────────────────────────────

# HOME
if page == "🏠 Home":
    st.markdown(
        '<div class="main-header">My AI Engineer Journey</div>',
        unsafe_allow_html=True
    )
    st.markdown("#### A production-grade AI assistant built from scratch — 30 days of Gen AI engineering")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-value">30</div>'
                    '<div class="metric-label">Days of learning</div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-value">10+</div>'
                    '<div class="metric-label">AI features built</div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-value">{docs.get("count", 0)}</div>'
                    '<div class="metric-label">Documents indexed</div>',
                    unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-value">100%</div>'
                    '<div class="metric-label">Local & free</div>',
                    unsafe_allow_html=True)

    st.divider()

    features = [
        ("💬", "Multi-Turn Chat",
         "Conversational AI with 4 personas and persistent memory"),
        ("📚", "RAG Q&A",
         "Document-grounded answers with source attribution"),
        ("📝", "Smart Summarizer",
         "Zero-shot, few-shot, and chain-of-thought summarization"),
        ("🔍", "Data Extractor",
         "Extract entities, contacts, meetings, and reviews"),
        ("🧠", "LLM Agents",
         "ReAct agents with tools, memory, and multi-agent systems"),
        ("🛡️", "Guardrails",
         "Input validation, PII detection, and output filtering"),
    ]

    col1, col2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<strong>{icon} {title}</strong><br>'
                f'<span style="color:#6B7280">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.divider()
    st.markdown("### 🛠 Tech Stack")
    tech = ["Python 3.11", "Ollama + llama3.2", "ChromaDB",
            "LangChain", "FastAPI", "Streamlit", "Docker"]
    cols = st.columns(len(tech))
    for col, t in zip(cols, tech):
        col.markdown(f"`{t}`")


# CHAT
elif page == "💬 Chat":
    st.markdown("## 💬 AI Chat")
    st.markdown("Multi-turn conversation with persona selection")

    col1, col2 = st.columns([3, 1])
    with col2:
        persona = st.selectbox(
            "Persona",
            ["default", "mentor", "socratic", "pirate"]
        )
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                result = chat_api(prompt, persona, history)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                reply = result.get("reply", "")
                st.markdown(reply)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )
                tokens = result.get("tokens_estimated", 0)
                st.caption(f"Tokens: ~{tokens} | Persona: {persona}")


# RAG Q&A
elif page == "📚 RAG Q&A":
    st.markdown("## 📚 Document Q&A")
    st.markdown("Ask questions grounded in your knowledge base")

    docs_info = get_documents()
    doc_count = docs_info.get("count", 0)

    if doc_count == 0:
        st.warning(
            "No documents indexed yet. "
            "Add .txt or .pdf files to `data/docs/` and run the ingestion pipeline."
        )
    else:
        st.success(f"✅ {doc_count} document chunks indexed and ready")

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Your question",
            placeholder="What is RAG and how does it work?"
        )
    with col2:
        top_k = st.slider("Sources", 1, 5, 3)

    if st.button("🔍 Ask", type="primary") and question:
        with st.spinner("Searching and generating answer..."):
            result = rag_api(question, top_k)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.markdown("### Answer")
            st.markdown(result.get("answer", ""))

            sources = result.get("sources", [])
            if sources:
                st.divider()
                st.markdown("### Sources Used")
                for i, src in enumerate(sources, 1):
                    with st.expander(
                        f"Source {i}: {src.get('source', 'unknown')} "
                        f"(score: {src.get('score', 0):.3f})"
                    ):
                        st.text(src.get("text", ""))

            st.caption(
                f"Retrieved {result.get('chunks_retrieved', 0)} chunks"
            )


# SUMMARIZER
elif page == "📝 Summarizer":
    st.markdown("## 📝 Smart Summarizer")
    st.markdown("Summarize any text using different prompting strategies")

    col1, col2 = st.columns([3, 1])
    with col2:
        style = st.selectbox(
            "Strategy",
            ["zero", "few", "cot"],
            format_func=lambda x: {
                "zero": "Zero-Shot",
                "few": "Few-Shot",
                "cot": "Chain-of-Thought"
            }[x]
        )

    text_input = st.text_area(
        "Text to summarize",
        placeholder="Paste any text here...",
        height=200
    )

    if st.button("✨ Summarize", type="primary") and text_input:
        with st.spinner(f"Summarizing with {style} strategy..."):
            result = summarize_api(text_input, style)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.markdown("### Summary")
            st.markdown(result.get("summary", ""))
            col1, col2, col3 = st.columns(3)
            col1.metric("Strategy", result.get("style", ""))
            col2.metric("Original", f"{result.get('original_length', 0)} chars")
            col3.metric("Summary", f"{result.get('summary_length', 0)} chars")


# DATA EXTRACTOR
elif page == "🔍 Data Extractor":
    st.markdown("## 🔍 Data Extractor")
    st.markdown("Extract structured information from unstructured text")

    col1, col2 = st.columns([3, 1])
    with col2:
        extract_type = st.selectbox(
            "Extract Type",
            ["entities", "contact", "meeting", "review"],
            format_func=lambda x: {
                "entities": "Named Entities",
                "contact": "Contact Info",
                "meeting": "Meeting Notes",
                "review": "Product Review"
            }[x]
        )

    samples = {
        "entities": "Apple CEO Tim Cook announced a new research center in Bangalore, India by March 2025.",
        "contact": "Hi, I'm Sarah Johnson, Senior PM at TechCorp. Email: sarah@techcorp.com, Phone: +1-555-0192",
        "meeting": "Q3 Planning Meeting — Oct 15. Attendees: Roshini, Priya, Arjun. Decision: launch by Nov 1st.",
        "review": "Amazing headphones! Best noise cancellation ever. Battery lasts forever. Bit pricey at Rs.29000 but worth it. 5/5 stars."
    }

    text_input = st.text_area(
        "Text to extract from",
        value=samples.get(extract_type, ""),
        height=150
    )

    if st.button("🔍 Extract", type="primary") and text_input:
        with st.spinner("Extracting structured data..."):
            result = extract_api(text_input, extract_type)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        elif result.get("success"):
            st.markdown("### Extracted Data")
            extracted = result.get("result", {})
            for key, value in extracted.items():
                if value:
                    col1, col2 = st.columns([1, 3])
                    col1.markdown(f"**{key.replace('_', ' ').title()}**")
                    col2.markdown(
                        str(value)[:200] if not isinstance(value, list)
                        else ", ".join(str(v) for v in value[:5])
                    )
        else:
            st.warning("Extraction failed. Try different text.")


# DASHBOARD
elif page == "📊 Dashboard":
    st.markdown("## 📊 System Dashboard")

    health = check_api_health()

    col1, col2, col3 = st.columns(3)
    with col1:
        status = health.get("status", "unknown")
        st.metric(
            "API Status",
            status.upper(),
            delta="healthy" if status == "healthy" else None
        )
    with col2:
        st.metric("Model", health.get("model", "unknown"))
    with col3:
        st.metric("Version", health.get("version", "unknown"))

    st.divider()
    st.markdown("### Features Available")
    features = health.get("features", [])
    cols = st.columns(len(features) if features else 1)
    for col, feature in zip(cols, features):
        col.success(f"✅ {feature}")

    st.divider()
    st.markdown("### Knowledge Base")
    docs_info = get_documents()
    documents = docs_info.get("documents", [])
    doc_count = docs_info.get("count", 0)

    st.metric("Total Chunks Indexed", doc_count)
    if documents:
        st.markdown("**Indexed Documents:**")
        for doc in documents[:10]:
            source = doc.get("source", "unknown")
            chunks = doc.get("chunks", 0)
            st.markdown(f"  - `{source}` — {chunks} chunks")

    st.divider()
    st.markdown("### API Endpoints")
    endpoints = [
        ("GET", "/health", "Health check — no auth required"),
        ("POST", "/chat/", "Chat with AI assistant"),
        ("GET", "/chat/personas", "List available personas"),
        ("POST", "/rag/ask", "RAG question answering"),
        ("GET", "/rag/documents", "List indexed documents"),
        ("POST", "/rag/summarize", "Text summarization"),
        ("POST", "/rag/extract", "Structured data extraction"),
    ]

    for method, path, desc in endpoints:
        col1, col2, col3 = st.columns([1, 2, 3])
        col1.code(method, language=None)
        col2.code(path, language=None)
        col3.markdown(desc)

    st.divider()
    if st.button("🔄 Refresh Dashboard"):
        st.rerun()