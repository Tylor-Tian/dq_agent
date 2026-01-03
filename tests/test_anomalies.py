import pandas as pd

from dq_agent.anomalies.detectors import missing_rate, outlier_mad


def test_missing_rate_detector_fails_on_high_null_rate():
    df = pd.DataFrame({"value": [1, None, 2, None]})
    result = missing_rate("value", df["value"], {"max_rate": 0.3}, sample_rows=10)

    assert result.status == "FAIL"
    assert result.metric["null_count"] == 2
    assert result.metric["null_rate"] == 0.5
    assert result.samples, "Expected samples for missing values"


def test_outlier_mad_detector_flags_outlier():
    df = pd.DataFrame({"value": [10, 11, 9, 10, 100]})
    result = outlier_mad("value", df["value"], {"z": 6.0}, sample_rows=5)

    assert result.status == "FAIL"
    assert result.metric["max_z"] > 6.0
    assert any(sample["value"] == 100 for sample in result.samples)
