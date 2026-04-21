import time
import json
import os
import ollama
from datetime import datetime
from observability.tracer import Tracer
from observability.metrics import MetricsCollector
from observability.alerts import AlertManager
from config import MODEL


class ObservabilityDashboard:
    """
    Full observability system — traces, metrics, and alerts in one place.

    Usage:
    dashboard = ObservabilityDashboard()

    # Wrap any AI call
    with dashboard.observe("chat", query="what is ML?") as ctx:
        result = your_ai_function(...)
        ctx["tokens"] = count_tokens(result)

    # Check system health
    dashboard.print_dashboard()
    """

    def __init__(self):
        self.tracer = Tracer()
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()

    def observe_chat(
        self,
        message: str,
        persona: str = "default",
        stream: bool = False
    ) -> dict:
        """Observe a chat request end to end."""
        with self.tracer.trace("chat", query=message) as trace:

            # Span 1: Prepare messages
            with self.tracer.span(trace, "prepare_messages"):
                from config import TEMPERATURE, MAX_TOKENS
                PERSONAS = {
                    "default": "You are a helpful AI assistant.",
                    "mentor": "You are a senior ML engineer mentoring a student."
                }
                system_prompt = PERSONAS.get(persona, PERSONAS["default"])
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]

            # Span 2: LLM call
            llm_start = time.time()
            with self.tracer.span(trace, "llm_generate", {"model": MODEL}):
                if stream:
                    print("🤖 Response: ", end="", flush=True)
                    full_reply = ""
                    for chunk in ollama.chat(
                        model=MODEL,
                        messages=messages,
                        stream=True,
                        options={"temperature": TEMPERATURE}
                    ):
                        token = chunk["message"]["content"]
                        print(token, end="", flush=True)
                        full_reply += token
                    print()
                    response_text = full_reply
                else:
                    response = ollama.chat(
                        model=MODEL,
                        messages=messages,
                        stream=False,
                        options={"temperature": TEMPERATURE}
                    )
                    response_text = response["message"]["content"]

            llm_ms = round((time.time() - llm_start) * 1000, 2)

            # Record metrics
            tokens_in = len(message.split())
            tokens_out = len(response_text.split())

            self.metrics.record_llm_call(
                request_type="chat",
                latency_ms=trace.total_duration_ms or llm_ms,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                model=MODEL,
                success=True,
                llm_ms=llm_ms
            )

        return {
            "response": response_text,
            "trace_id": trace.trace_id,
            "llm_latency_ms": llm_ms
        }

    def observe_rag(self, question: str, top_k: int = 3) -> dict:
        """Observe a RAG request with separate retrieval and generation spans."""
        with self.tracer.trace("rag", query=question) as trace:

            # Span 1: Embed query
            with self.tracer.span(trace, "embed_query"):
                import ollama as ol
                response = ol.embeddings(
                    model="nomic-embed-text",
                    prompt=question
                )
                query_embedding = response["embedding"]

            # Span 2: Vector search
            retrieval_start = time.time()
            with self.tracer.span(trace, "vector_search", {"top_k": top_k}):
                from rag.knowledge_base import search
                chunks = search(query=question, top_k=top_k)

            retrieval_ms = round((time.time() - retrieval_start) * 1000, 2)

            if not chunks:
                return {
                    "answer": "No relevant information found.",
                    "trace_id": trace.trace_id,
                    "chunks_found": 0
                }

            # Span 3: Build context
            with self.tracer.span(trace, "build_context"):
                context = "\n\n---\n\n".join(
                    f"[Source: {c['source']}]\n{c['text']}"
                    for c in chunks
                )
                prompt = f"""Answer using ONLY the context below. Cite your source.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

            # Span 4: LLM generation
            llm_start = time.time()
            print("🤖 Answer: ", end="", flush=True)
            full_reply = ""

            with self.tracer.span(trace, "llm_generate", {"model": MODEL}):
                for chunk in ollama.chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    options={"temperature": 0.3}
                ):
                    token = chunk["message"]["content"]
                    print(token, end="", flush=True)
                    full_reply += token
                print()

            llm_ms = round((time.time() - llm_start) * 1000, 2)

            # Record metrics
            self.metrics.record_llm_call(
                request_type="rag",
                latency_ms=trace.total_duration_ms or (retrieval_ms + llm_ms),
                tokens_input=len(prompt.split()),
                tokens_output=len(full_reply.split()),
                model=MODEL,
                success=True,
                retrieval_ms=retrieval_ms,
                llm_ms=llm_ms,
                metadata={"chunks_found": len(chunks)}
            )

        return {
            "answer": full_reply,
            "trace_id": trace.trace_id,
            "chunks_found": len(chunks),
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms
        }

    def print_dashboard(self):
        """Print a full observability dashboard."""
        print(f"\n{'='*60}")
        print(f"  OBSERVABILITY DASHBOARD")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        all_stats = self.metrics.get_stats()
        if "error" not in all_stats:
            print(f"\n📊 OVERALL METRICS ({all_stats['total_requests']} requests)")
            print(f"  Latency p50:   {all_stats['latency_p50_ms']}ms")
            print(f"  Latency p95:   {all_stats['latency_p95_ms']}ms")
            print(f"  Latency p99:   {all_stats['latency_p99_ms']}ms")
            print(f"  Error rate:    {all_stats['error_rate_pct']}%")
            print(f"  Cache hit rate:{all_stats['cache_hit_rate_pct']}%")
            print(f"  Total tokens:  {all_stats['total_tokens']}")
            print(f"  Total cost:    ${all_stats['total_cost_usd']:.6f}")

        for req_type in ["chat", "rag"]:
            stats = self.metrics.get_stats(request_type=req_type)
            if "error" not in stats and stats["total_requests"] > 0:
                print(f"\n  [{req_type.upper()}] {stats['total_requests']} requests")
                print(f"    Avg latency: {stats['latency_avg_ms']}ms")
                print(f"    Error rate:  {stats['error_rate_pct']}%")
                print(f"    Avg tokens:  {stats['avg_tokens_per_request']}")

        print(f"\n🔔 ALERTS")
        triggered = self.alerts.evaluate(all_stats)
        if not triggered:
            print("  ✅ All systems healthy — no alerts")
        else:
            for alert in triggered:
                icon = "🔴" if alert.severity == "critical" else "🟡"
                print(f"  {icon} {alert.message}")

        print(f"\n📝 RECENT TRACES")
        self.tracer.print_recent(limit=5)

        alert_summary = self.alerts.summary()
        print(f"\n  Total alerts fired: {alert_summary['total_alerts']}")
        print(f"  Critical: {alert_summary['critical']} | "
              f"Warning: {alert_summary['warning']}")
        print(f"{'='*60}\n")

    def run_health_check(self) -> dict:
        """
        Run a quick health check by making test requests.
        Returns system health status.
        """
        print("\n🏥 Running health check...")
        results = {}

        # Test chat
        try:
            start = time.time()
            result = self.observe_chat("What is 2 + 2?")
            results["chat"] = {
                "status": "healthy",
                "latency_ms": round((time.time() - start) * 1000, 2)
            }
            print("  ✅ Chat: healthy")
        except Exception as e:
            results["chat"] = {"status": "error", "error": str(e)}
            print(f"  ❌ Chat: {e}")

        # Test RAG
        try:
            start = time.time()
            result = self.observe_rag("what is machine learning")
            results["rag"] = {
                "status": "healthy" if result["chunks_found"] > 0 else "degraded",
                "latency_ms": round((time.time() - start) * 1000, 2),
                "chunks_found": result["chunks_found"]
            }
            status = "healthy" if result["chunks_found"] > 0 else "degraded"
            print(f"  {'✅' if status == 'healthy' else '🟡'} RAG: {status}")
        except Exception as e:
            results["rag"] = {"status": "error", "error": str(e)}
            print(f"  ❌ RAG: {e}")

        overall = "healthy" if all(
            r.get("status") == "healthy" for r in results.values()
        ) else "degraded"

        print(f"\n  Overall: {'✅ healthy' if overall == 'healthy' else '🟡 degraded'}")
        return {"overall": overall, "services": results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Observability Dashboard")
    parser.add_argument("--health",    action="store_true",
                        help="Run health check")
    parser.add_argument("--dashboard", action="store_true",
                        help="Show dashboard")
    parser.add_argument("--chat",      type=str,
                        help="Observed chat request")
    parser.add_argument("--rag",       type=str,
                        help="Observed RAG request")
    parser.add_argument("--simulate",  action="store_true",
                        help="Simulate requests to generate metrics")
    args = parser.parse_args()

    dashboard = ObservabilityDashboard()

    if args.health:
        dashboard.run_health_check()
        dashboard.print_dashboard()

    elif args.chat:
        result = dashboard.observe_chat(args.chat, stream=True)
        print(f"\n[Trace ID: {result['trace_id']} | "
              f"LLM: {result['llm_latency_ms']}ms]")

    elif args.rag:
        result = dashboard.observe_rag(args.rag)
        print(f"\n[Trace ID: {result['trace_id']} | "
              f"Retrieval: {result['retrieval_ms']}ms | "
              f"LLM: {result['llm_ms']}ms | "
              f"Chunks: {result['chunks_found']}]")

    elif args.simulate:
        print("Simulating requests to generate metrics...")
        test_queries = [
            "What is machine learning?",
            "Explain neural networks.",
            "What is RAG?",
            "How do embeddings work?",
            "What is deep learning?"
        ]
        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}/{len(test_queries)}] Simulating: {query}")
            dashboard.observe_chat(query)
        dashboard.print_dashboard()

    elif args.dashboard:
        dashboard.print_dashboard()

    else:
        print("Running health check and showing dashboard...")
        dashboard.run_health_check()
        dashboard.print_dashboard()