import time
import json
import os
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field


METRICS_LOG_DIR = "observability/metrics"

# Cost estimates (per 1000 tokens) — adjust for your setup
# For Ollama local models, cost is compute time not money
COST_PER_1K_TOKENS = {
    "llama3.2": 0.0,          # Free — local model
    "nomic-embed-text": 0.0,  # Free — local model
    "gpt-3.5-turbo": 0.002,   # If using OpenAI
    "gpt-4": 0.03,            # If using OpenAI
}

# Time cost — electricity/compute estimate ($/hour for GPU)
COMPUTE_COST_PER_HOUR = 0.50  # rough estimate for local GPU


@dataclass
class RequestMetric:
    """Metrics for a single request."""
    request_id: str
    request_type: str
    timestamp: str
    latency_ms: float
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    model: str = "llama3.2"
    estimated_cost_usd: float = 0.0
    success: bool = True
    error_type: str = ""
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    cache_hit: bool = False
    metadata: dict = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates metrics for all AI requests.

    Tracks:
    - Request latency (p50, p95, p99)
    - Token usage and costs
    - Error rates by request type
    - Cache hit rates
    - Requests per minute
    """

    def __init__(self, log_dir: str = METRICS_LOG_DIR):
        self.log_dir = log_dir
        self.metrics: list[RequestMetric] = []
        self.counters = defaultdict(int)
        os.makedirs(log_dir, exist_ok=True)

    def record(self, metric: RequestMetric):
        """Record a single request metric."""
        self.metrics.append(metric)
        self.counters[f"requests_{metric.request_type}"] += 1
        self.counters["requests_total"] += 1

        if not metric.success:
            self.counters[f"errors_{metric.request_type}"] += 1
            self.counters["errors_total"] += 1

        if metric.cache_hit:
            self.counters["cache_hits"] += 1

        self._save_metric(metric)

    def record_llm_call(
        self,
        request_type: str,
        latency_ms: float,
        tokens_input: int = 0,
        tokens_output: int = 0,
        model: str = "llama3.2",
        success: bool = True,
        error_type: str = "",
        retrieval_ms: float = 0.0,
        llm_ms: float = 0.0,
        cache_hit: bool = False,
        metadata: dict = None
    ) -> RequestMetric:
        """Record metrics for an LLM call."""
        import uuid
        tokens_total = tokens_input + tokens_output
        cost = self._estimate_cost(tokens_total, model, latency_ms)

        metric = RequestMetric(
            request_id=str(uuid.uuid4())[:8],
            request_type=request_type,
            timestamp=datetime.now().isoformat(),
            latency_ms=latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_total=tokens_total,
            model=model,
            estimated_cost_usd=cost,
            success=success,
            error_type=error_type,
            retrieval_latency_ms=retrieval_ms,
            llm_latency_ms=llm_ms,
            cache_hit=cache_hit,
            metadata=metadata or {}
        )

        self.record(metric)
        return metric

    def _estimate_cost(
        self,
        tokens: int,
        model: str,
        latency_ms: float
    ) -> float:
        """Estimate cost for a request."""
        # API cost
        cost_per_1k = COST_PER_1K_TOKENS.get(model, 0.002)
        api_cost = (tokens / 1000) * cost_per_1k

        # Compute cost (for local models)
        compute_cost = (latency_ms / 3_600_000) * COMPUTE_COST_PER_HOUR

        return round(api_cost + compute_cost, 6)

    def get_stats(
        self,
        request_type: str = None,
        last_n: int = None
    ) -> dict:
        """Calculate aggregate statistics."""
        metrics = self.metrics

        if request_type:
            metrics = [m for m in metrics if m.request_type == request_type]

        if last_n:
            metrics = metrics[-last_n:]

        if not metrics:
            return {"error": "No metrics available"}

        latencies = [m.latency_ms for m in metrics]
        latencies.sort()

        total_tokens = sum(m.tokens_total for m in metrics)
        total_cost = sum(m.estimated_cost_usd for m in metrics)
        errors = [m for m in metrics if not m.success]
        cache_hits = [m for m in metrics if m.cache_hit]

        def percentile(data: list, p: int) -> float:
            if not data:
                return 0.0
            k = int(len(data) * p / 100)
            return round(data[min(k, len(data)-1)], 2)

        return {
            "total_requests": len(metrics),
            "request_type": request_type or "all",
            "latency_p50_ms": percentile(latencies, 50),
            "latency_p95_ms": percentile(latencies, 95),
            "latency_p99_ms": percentile(latencies, 99),
            "latency_avg_ms": round(sum(latencies) / len(latencies), 2),
            "latency_max_ms": round(max(latencies), 2),
            "error_rate_pct": round(len(errors) / len(metrics) * 100, 2),
            "error_count": len(errors),
            "cache_hit_rate_pct": round(len(cache_hits) / len(metrics) * 100, 2),
            "total_tokens": total_tokens,
            "avg_tokens_per_request": round(total_tokens / len(metrics), 1),
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_per_request_usd": round(total_cost / len(metrics), 6),
        }

    def _save_metric(self, metric: RequestMetric):
        """Save metric to disk."""
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(
            self.log_dir, f"metrics_{date_str}.jsonl"
        )
        with open(log_file, "a") as f:
            f.write(json.dumps({
                "request_id": metric.request_id,
                "request_type": metric.request_type,
                "timestamp": metric.timestamp,
                "latency_ms": metric.latency_ms,
                "tokens_total": metric.tokens_total,
                "estimated_cost_usd": metric.estimated_cost_usd,
                "success": metric.success,
                "cache_hit": metric.cache_hit,
                "model": metric.model
            }) + "\n")