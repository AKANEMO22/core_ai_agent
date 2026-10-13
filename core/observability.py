"""Observability primitives: safe session references, structured JSON logging, metrics registry.

Metrics are exposed in the Prometheus text exposition format without an external dependency.
"""

import contextvars
import hashlib
import json
import logging
import math
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)

DEFAULT_BUCKETS: Tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0, 10.0, 30.0, 60.0)


def session_ref(session_id: Optional[str]) -> str:
    """Non-reversible 12-character session reference safe for logs and metrics labels."""
    if not session_id:
        return "-"
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line with level, logger, message, correlation id and extras.

    Callers are responsible for never passing candidate content into log messages; the
    formatter adds no request data of its own.
    """

    RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        correlation_id = correlation_id_var.get()
        if correlation_id:
            entry["correlation_id"] = correlation_id
        for key, value in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(entry, default=str)


_SESSION_ID_PATTERN = re.compile(r"\bsess_[A-Za-z0-9_-]{6,64}\b")


class SessionIdRedactionFilter(logging.Filter):
    """Replaces generated session ids (sess_*) in log messages with their non-reversible reference.

    Defense in depth for error strings that embed an id; application code logs `session_ref`.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "sess_" in message:
            record.msg = _SESSION_ID_PATTERN.sub(lambda m: f"session:{session_ref(m.group(0))}", message)
            record.args = None
        return True


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Install the structured formatter and redaction filter on the root logger (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        if getattr(handler, "_ai_runtime_handler", False):
            return
    handler = logging.StreamHandler()
    handler._ai_runtime_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(JsonLogFormatter() if json_format else logging.Formatter())
    handler.addFilter(SessionIdRedactionFilter())
    root.addHandler(handler)


def _label_key(labels: Optional[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((labels or {}).items()))


def _format_labels(labels: Tuple[Tuple[str, str], ...], extra: Optional[Tuple[str, str]] = None) -> str:
    items = list(labels) + ([extra] if extra else [])
    if not items:
        return ""
    rendered = ",".join(f'{k}="{str(v).replace(chr(34), chr(39))}"' for k, v in items)
    return "{" + rendered + "}"


class MetricsRegistry:
    """Thread-safe counters, gauges and histograms rendered in Prometheus text format."""

    def __init__(self, namespace: str = "ai_interview", buckets: Sequence[float] = DEFAULT_BUCKETS) -> None:
        self.namespace = namespace
        self.buckets = tuple(sorted(buckets))
        self._lock = threading.Lock()
        self._counters: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = {}
        self._gauges: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = {}
        self._histograms: Dict[str, Dict[Tuple[Tuple[str, str], ...], Dict[str, Any]]] = {}
        self._help: Dict[str, str] = {}

    def describe(self, name: str, help_text: str) -> None:
        """Attach HELP text to a metric."""
        self._help[name] = help_text

    def inc(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increase a counter."""
        with self._lock:
            series = self._counters.setdefault(name, {})
            key = _label_key(labels)
            series[key] = series.get(key, 0.0) + value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge value."""
        with self._lock:
            self._gauges.setdefault(name, {})[_label_key(labels)] = float(value)

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram observation."""
        with self._lock:
            series = self._histograms.setdefault(name, {})
            key = _label_key(labels)
            hist = series.get(key)
            if hist is None:
                hist = {"buckets": [0] * len(self.buckets), "count": 0, "sum": 0.0, "values": []}
                series[key] = hist
            for idx, bound in enumerate(self.buckets):
                if value <= bound:
                    hist["buckets"][idx] += 1
            hist["count"] += 1
            hist["sum"] += value
            hist["values"].append(value)
            if len(hist["values"]) > 10000:
                del hist["values"][:5000]

    def counter_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Current counter value (tests and health)."""
        return self._counters.get(name, {}).get(_label_key(labels), 0.0)

    def quantile(self, name: str, q: float, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Approximate quantile from retained recent observations."""
        hist = self._histograms.get(name, {}).get(_label_key(labels))
        if not hist or not hist["values"]:
            return None
        values = sorted(hist["values"])
        index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
        return values[index]

    def render(self) -> str:
        """Render all metrics in the Prometheus text exposition format."""
        lines: List[str] = []
        with self._lock:
            for name, series in sorted(self._counters.items()):
                full = f"{self.namespace}_{name}"
                if name in self._help:
                    lines.append(f"# HELP {full} {self._help[name]}")
                lines.append(f"# TYPE {full} counter")
                for labels, value in series.items():
                    lines.append(f"{full}{_format_labels(labels)} {value}")
            for name, series in sorted(self._gauges.items()):
                full = f"{self.namespace}_{name}"
                if name in self._help:
                    lines.append(f"# HELP {full} {self._help[name]}")
                lines.append(f"# TYPE {full} gauge")
                for labels, value in series.items():
                    lines.append(f"{full}{_format_labels(labels)} {value}")
            for name, series in sorted(self._histograms.items()):
                full = f"{self.namespace}_{name}"
                if name in self._help:
                    lines.append(f"# HELP {full} {self._help[name]}")
                lines.append(f"# TYPE {full} histogram")
                for labels, hist in series.items():
                    for bound, count in zip(self.buckets, hist["buckets"]):
                        lines.append(f"{full}_bucket{_format_labels(labels, ('le', str(bound)))} {count}")
                    lines.append(f"{full}_bucket{_format_labels(labels, ('le', '+Inf'))} {hist['count']}")
                    lines.append(f"{full}_sum{_format_labels(labels)} {hist['sum']}")
                    lines.append(f"{full}_count{_format_labels(labels)} {hist['count']}")
        return "\n".join(lines) + "\n"


class Timer:
    """Context manager observing elapsed seconds into a histogram."""

    def __init__(self, registry: Optional[MetricsRegistry], name: str, labels: Optional[Dict[str, str]] = None) -> None:
        self.registry = registry
        self.name = name
        self.labels = labels
        self.started = 0.0

    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.registry is not None:
            self.registry.observe(self.name, time.perf_counter() - self.started, self.labels)


def build_default_metrics() -> MetricsRegistry:
    """Registry with HELP text for every metric the runtime emits."""
    registry = MetricsRegistry()
    for name, help_text in {
        "ttft_seconds": "Time from accepted answer (including queue wait) to first response token",
        "turn_duration_seconds": "Duration of an interviewer turn",
        "busy_total": "Requests refused with busy",
        "provider_errors_total": "LLM provider errors by agent",
        "assessment_lag_seconds": "Delay between answer and persisted evaluation",
        "assessment_pending": "Jobs pending in the assessment stream",
        "dead_letters_total": "Assessment jobs moved to the dead-letter stream",
        "compactions_total": "Context compactions by agent profile",
        "llm_tokens_total": "LLM tokens by agent and kind",
        "llm_cost_usd_total": "Estimated LLM spend by agent",
        "active_connections": "Active WebSocket connection leases",
        "directive_ready_total": "Orchestrator directives by source",
        "orchestrator_cycle_seconds": "Duration of an orchestrator cycle",
        "degraded_total": "Degraded-mode activations by component",
        "speaker_hedge_total": "Hedged speaker requests by winning racer",
    }.items():
        registry.describe(name, help_text)
    return registry
