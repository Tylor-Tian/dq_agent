"""Rule engine."""

from dq_agent.rules import checks as _checks
from dq_agent.rules.base import RuleResult, run_rules

__all__ = ["RuleResult", "run_rules"]
