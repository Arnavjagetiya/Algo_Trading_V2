import pandas as pd

from analytics.metrics import PerformanceMetrics


def test_metrics_calculate_returns_summary_dict():
    metrics = PerformanceMetrics()
    equity_curve = pd.Series([100.0, 105.0, 98.0], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    returns = equity_curve.pct_change().fillna(0.0)

    result = metrics.calculate(equity_curve=equity_curve, returns=returns, trades=[])

    assert isinstance(result, dict)
    assert "total_return" in result
    assert "annual_return" in result
    assert "cagr" in result
