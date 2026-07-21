import pandas as pd

from models.optimizer import PortfolioOptimizer


def test_optimizer_optimize_accepts_price_frame():
    prices = pd.DataFrame(
        {
            "A": [100 + i * 0.5 for i in range(260)],
            "B": [110 + i * 0.4 for i in range(260)],
        },
        index=pd.date_range("2024-01-01", periods=260, freq="D"),
    )

    optimizer = PortfolioOptimizer()
    result = optimizer.optimize(prices)

    assert result is not None
    assert hasattr(result, "weights")
    assert result.method in {"Maximum Sharpe", "Minimum Variance", "Equal Weight", "Hierarchical Risk Parity", "Risk Parity"}
