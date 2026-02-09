import pandas as pd

from dq_agent.config import ColumnConfig, Config, ReportConfig
from dq_agent.rules import run_rules


def _run_single_rule(df: pd.DataFrame, column: str, check: dict) -> dict:
    cfg = Config(columns={column: ColumnConfig(checks=[check])}, report=ReportConfig(sample_rows=5))
    results = run_rules(df, cfg)
    assert len(results) == 1
    return results[0].to_dict()


def test_not_null_rule_pass_and_fail():
    df = pd.DataFrame({"user_id": ["a", None, "b", "c"]})
    passing = _run_single_rule(df, "user_id", {"not_null": {"max_null_rate": 0.5}})
    failing = _run_single_rule(df, "user_id", {"not_null": {"max_null_rate": 0.1}})
    assert passing["status"] == "PASS"
    assert failing["status"] == "FAIL"
    assert failing["failed_count"] == 1


def test_unique_rule_pass_and_fail():
    df_pass = pd.DataFrame({"order_id": ["1", "2", "3"]})
    df_fail = pd.DataFrame({"order_id": ["1", "2", "2", "3"]})
    passing = _run_single_rule(df_pass, "order_id", {"unique": True})
    failing = _run_single_rule(df_fail, "order_id", {"unique": True})
    assert passing["status"] == "PASS"
    assert failing["failed_count"] == 2


def test_range_rule_pass_and_fail():
    df = pd.DataFrame({"amount": [10, 20, -5, 50]})
    passing = _run_single_rule(df.head(2), "amount", {"range": {"min": 0, "max": 100}})
    failing = _run_single_rule(df, "amount", {"range": {"min": 0, "max": 100}})
    assert passing["status"] == "PASS"
    assert failing["status"] == "FAIL"
    assert failing["failed_count"] == 1


def test_allowed_values_rule_pass_and_fail():
    df = pd.DataFrame({"status": ["PAID", "REFUND", "UNKNOWN"]})
    passing = _run_single_rule(df.head(2), "status", {"allowed_values": {"values": ["PAID", "REFUND"]}})
    failing = _run_single_rule(df, "status", {"allowed_values": {"values": ["PAID", "REFUND"]}})
    assert passing["status"] == "PASS"
    assert failing["status"] == "FAIL"
    assert failing["failed_count"] == 1


def test_string_noise_rule_detects_literal_patterns():
    df = pd.DataFrame({"txt": ["ok", "bad*", "also '' bad", None]})
    result = _run_single_rule(
        df,
        "txt",
        {"string_noise": {"contains": ["*", "''"], "max_rate": 0.0}},
    )

    assert result["status"] == "FAIL"
    # only 2 non-null values match the patterns
    assert result["failed_count"] == 2
    assert any(s["value"] == "bad*" for s in result["samples"])
