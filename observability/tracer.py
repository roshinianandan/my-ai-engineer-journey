import time
import uuid
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from contextlib import contextmanager


TRACE_LOG_DIR = "observability/traces"


@dataclass
class Span:
    """
    A single operation within a trace.
    Traces are made up of spans — each span is one step.

    Example trace for a RAG request:
    - Span 1: embed_query (5ms)
    - Span 2: vector_search (12ms)
    - Span 3: llm_generate (2400ms)
    - Span 4: format_response (1ms)
    Total trace: 2418ms
    """
    span_id: str
    trace_id: str
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "running"       # running, success, error
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    def finish(self, status: str = "success", metadata: dict = None):
        """Mark the span as complete."""
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.status = status
        if metadata:
            self.metadata.update(metadata)

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "metadata": self.metadata,
            "error": self.error
        }


@dataclass
class Trace:
    """
    A complete end-to-end request trace.
    Contains all spans for a single user request.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    request_type: str = "unknown"
    query: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration_ms: Optional[float] = None
    spans: list = field(default_factory=list)
    status: str = "running"
    metadata: dict = field(default_factory=dict)

    def add_span(self, name: str, metadata: dict = None) -> Span:
        """Start a new span within this trace."""
        span = Span(
            span_id=str(uuid.uuid4())[:8],
            trace_id=self.trace_id,
            name=name,
            metadata=metadata or {}
        )
        self.spans.append(span)
        return span

    def finish(self, status: str = "success"):
        """Complete the trace."""
        self.end_time = time.time()
        self.total_duration_ms = round(
            (self.end_time - self.start_time) * 1000, 2
        )
        self.status = status

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "request_type": self.request_type,
            "query": self.query[:100],
            "start_time": self.start_time,
            "timestamp": datetime.fromtimestamp(self.start_time).isoformat(),
            "total_duration_ms": self.total_duration_ms,
            "status": self.status,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata
        }

    def summary(self) -> str:
        """Return a one-line summary of the trace."""
        spans_summary = " | ".join(
            f"{s.name}:{s.duration_ms}ms"
            for s in self.spans
            if s.duration_ms
        )
        return (
            f"[{self.trace_id}] {self.request_type} | "
            f"total:{self.total_duration_ms}ms | "
            f"status:{self.status} | {spans_summary}"
        )


class Tracer:
    """
    Distributed tracing system for AI requests.

    Records every operation with timing, input/output metadata,
    and error information. Essential for debugging production AI systems.
    """

    def __init__(self, log_dir: str = TRACE_LOG_DIR):
        self.log_dir = log_dir
        self.active_traces: dict[str, Trace] = {}
        self.completed_traces: list[Trace] = []
        os.makedirs(log_dir, exist_ok=True)

    def start_trace(
        self,
        request_type: str,
        query: str = "",
        metadata: dict = None
    ) -> Trace:
        """Start a new trace for a request."""
        trace = Trace(
            request_type=request_type,
            query=query,
            metadata=metadata or {}
        )
        self.active_traces[trace.trace_id] = trace
        print(f"\n[Tracer] Started trace {trace.trace_id}: {request_type}")
        return trace

    def finish_trace(self, trace: Trace, status: str = "success"):
        """Complete a trace and save it."""
        trace.finish(status=status)
        self.active_traces.pop(trace.trace_id, None)
        self.completed_traces.append(trace)
        self._save_trace(trace)
        print(f"[Tracer] {trace.summary()}")
        return trace

    @contextmanager
    def trace(self, request_type: str, query: str = "", metadata: dict = None):
        """Context manager for automatic trace lifecycle management."""
        t = self.start_trace(request_type, query, metadata)
        try:
            yield t
            self.finish_trace(t, status="success")
        except Exception as e:
            t.metadata["error"] = str(e)
            self.finish_trace(t, status="error")
            raise

    @contextmanager
    def span(self, trace: Trace, name: str, metadata: dict = None):
        """Context manager for automatic span lifecycle management."""
        s = trace.add_span(name, metadata)
        try:
            yield s
            s.finish(status="success")
        except Exception as e:
            s.error = str(e)
            s.finish(status="error")
            raise

    def _save_trace(self, trace: Trace):
        """Save trace to disk for persistence."""
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.log_dir, f"traces_{date_str}.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")

    def get_recent_traces(self, limit: int = 10) -> list:
        """Get the most recent completed traces."""
        return self.completed_traces[-limit:]

    def print_recent(self, limit: int = 5):
        """Print recent trace summaries."""
        traces = self.get_recent_traces(limit)
        print(f"\n{'='*60}")
        print(f"  RECENT TRACES (last {len(traces)})")
        print(f"{'='*60}")
        for t in traces:
            print(f"  {t.summary()}")
        print(f"{'='*60}\n")