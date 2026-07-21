import pandas as pd

from models.garch import GARCHModel


def test_garch_accepts_dataframe_with_close_column():
    prices = pd.DataFrame(
        {
            "Close": [100 + i * 0.1 for i in range(300)],
        },
        index=pd.date_range("2024-01-01", periods=300, freq="D"),
    )

    model = GARCHModel(window=60)
    result = model.fit(prices)

    assert result is not None
    assert "Predicted_Volatility" in result.data.columns
    assert "Return" in result.data.columns
