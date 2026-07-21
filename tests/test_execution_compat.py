import pandas as pd

from strategy.execution import ExecutionEngine


def test_execute_accepts_backtest_compat_kwargs():
    engine = ExecutionEngine()

    fill = engine.execute(
        symbol="AAPL",
        signal=1,
        quantity=10,
        price=100.0,
        timestamp=pd.Timestamp("2024-01-01"),
    )

    assert fill["symbol"] == "AAPL"
    assert fill["quantity"] == 10
    assert fill["signal"] == 1
    assert fill["price"] == 100.0
