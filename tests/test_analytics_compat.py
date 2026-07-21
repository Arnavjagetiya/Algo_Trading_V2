from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.plots import PerformancePlots
from analytics.tearsheet import TearSheet


def test_performance_plots_support_generate_all():
    plots = PerformancePlots()
    returns = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2024-01-01", periods=3, freq="D"))

    fig = plots.generate_all(equity_curve=returns, trades=pd.DataFrame(), portfolio=pd.DataFrame())

    assert fig is not None


def test_tearsheet_support_create_and_save():
    tearsheet = TearSheet()
    returns = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2024-01-01", periods=3, freq="D"))

    report = tearsheet.create(returns, pd.DataFrame(), pd.DataFrame())
    assert report is not None

    html = tearsheet.save(report, None)
    assert html is None
