import json
import os
from datetime import datetime
from dataclasses import dataclass


ALERTS_LOG_DIR = "observability/alerts"


@dataclass
class AlertRule:
    """A rule that triggers an alert when violated."""
    name: str
    metric: str
    threshold: float
    operator: str      # gt, lt, gte, lte
    severity: str      # warning, critical
    message_template: str


@dataclass
class Alert:
    """A triggered alert."""
    rule_name: str
    severity: str
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: str


# Default alert rules
DEFAULT_RULES = [
    AlertRule(
        name="high_latency",
        metric="latency_p95_ms",
        threshold=10000,
        operator="gt",
        severity="warning",
        message_template="P95 latency {value}ms exceeds threshold {threshold}ms"
    ),
    AlertRule(
        name="critical_latency",
        metric="latency_p99_ms",
        threshold=30000,
        operator="gt",
        severity="critical",
        message_template="P99 latency {value}ms critically high (threshold {threshold}ms)"
    ),
    AlertRule(
        name="high_error_rate",
        metric="error_rate_pct",
        threshold=10.0,
        operator="gt",
        severity="warning",
        message_template="Error rate {value}% exceeds threshold {threshold}%"
    ),
    AlertRule(
        name="critical_error_rate",
        metric="error_rate_pct",
        threshold=25.0,
        operator="gt",
        severity="critical",
        message_template="Critical error rate {value}% — immediate attention required"
    ),
    AlertRule(
        name="high_cost",
        metric="total_cost_usd",
        threshold=1.0,
        operator="gt",
        severity="warning",
        message_template="Total cost ${value:.4f} exceeds budget threshold ${threshold}"
    ),
]


class AlertManager:
    """
    Evaluates metrics against rules and fires alerts.

    In production this would:
    - Send emails or Slack messages
    - Page on-call engineers via PagerDuty
    - Create tickets in Jira
    - Trigger auto-scaling

    Here we log to file and print to console.
    """

    def __init__(
        self,
        rules: list[AlertRule] = None,
        log_dir: str = ALERTS_LOG_DIR
    ):
        self.rules = rules or DEFAULT_RULES
        self.log_dir = log_dir
        self.fired_alerts: list[Alert] = []
        os.makedirs(log_dir, exist_ok=True)

    def add_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        self.rules.append(rule)

    def evaluate(self, stats: dict) -> list[Alert]:
        """
        Evaluate all rules against current metrics.
        Returns list of triggered alerts.
        """
        triggered = []

        for rule in self.rules:
            value = stats.get(rule.metric)
            if value is None:
                continue

            should_alert = False
            if rule.operator == "gt" and value > rule.threshold:
                should_alert = True
            elif rule.operator == "lt" and value < rule.threshold:
                should_alert = True
            elif rule.operator == "gte" and value >= rule.threshold:
                should_alert = True
            elif rule.operator == "lte" and value <= rule.threshold:
                should_alert = True

            if should_alert:
                alert = Alert(
                    rule_name=rule.name,
                    severity=rule.severity,
                    metric=rule.metric,
                    value=round(value, 4),
                    threshold=rule.threshold,
                    message=rule.message_template.format(
                        value=value,
                        threshold=rule.threshold
                    ),
                    timestamp=datetime.now().isoformat()
                )
                triggered.append(alert)
                self.fired_alerts.append(alert)
                self._fire_alert(alert)

        return triggered

    def _fire_alert(self, alert: Alert):
        """Fire an alert — log and display."""
        icon = "🔴" if alert.severity == "critical" else "🟡"
        print(f"\n{icon} ALERT [{alert.severity.upper()}]: {alert.message}")

        # Log to file
        log_file = os.path.join(
            self.log_dir,
            f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        with open(log_file, "a") as f:
            f.write(json.dumps({
                "rule": alert.rule_name,
                "severity": alert.severity,
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold,
                "message": alert.message,
                "timestamp": alert.timestamp
            }) + "\n")

    def get_active_alerts(self) -> list[Alert]:
        """Get all fired alerts."""
        return self.fired_alerts

    def summary(self) -> dict:
        """Return alert summary."""
        return {
            "total_alerts": len(self.fired_alerts),
            "critical": sum(1 for a in self.fired_alerts if a.severity == "critical"),
            "warning": sum(1 for a in self.fired_alerts if a.severity == "warning"),
            "rules_active": len(self.rules)
        }