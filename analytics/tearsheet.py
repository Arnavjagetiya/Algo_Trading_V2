"""
analytics/tearsheet.py
=========================================================
Professional Portfolio Tear Sheet
=========================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from analytics.metrics import PerformanceMetrics
from analytics.plots import PerformancePlots


class TearSheet:
    """
    Professional Portfolio Tear Sheet

    ✓ Performance Summary
    ✓ Risk Summary
    ✓ Benchmark Comparison
    ✓ Charts
    ✓ Trade Statistics
    """

    def __init__(

        self,

        trading_days: int = 252,

        risk_free_rate: float = 0.06,

    ):

        self.metrics = PerformanceMetrics(

            trading_days=trading_days,

            risk_free_rate=risk_free_rate,

        )

        self.plots = PerformancePlots()

        self.trading_days = trading_days

        self.risk_free_rate = risk_free_rate

    # ========================================================
    # PERFORMANCE SUMMARY
    # ========================================================

    def performance_summary(

        self,

        returns: pd.Series,

    ) -> pd.DataFrame:

        return self.metrics.summary(

            returns

        )

    # ========================================================
    # BENCHMARK SUMMARY
    # ========================================================

    def benchmark_summary(

        self,

        portfolio_returns: pd.Series,

        benchmark_returns: pd.Series,

    ) -> pd.DataFrame:

        return self.metrics.benchmark_summary(

            portfolio_returns,

            benchmark_returns,

        )

    # ========================================================
    # PERFORMANCE TABLE
    # ========================================================

    def performance_table(

        self,

        returns: pd.Series,

    ) -> pd.DataFrame:

        summary = self.performance_summary(

            returns

        )

        table = summary.T

        table.columns = [

            "Value"

        ]

        return table

    # ========================================================
    # RISK TABLE
    # ========================================================

    def risk_table(

        self,

        returns: pd.Series,

    ) -> pd.DataFrame:

        metrics = {

            "Volatility":

                self.metrics.volatility(

                    returns

                ),

            "Downside Volatility":

                self.metrics.downside_volatility(

                    returns

                ),

            "Maximum Drawdown":

                self.metrics.maximum_drawdown(

                    returns

                ),

            "VaR (95%)":

                self.metrics.value_at_risk(

                    returns

                ),

            "CVaR (95%)":

                self.metrics.conditional_value_at_risk(

                    returns

                ),

            "Ulcer Index":

                self.metrics.ulcer_index(

                    returns

                ),

        }

        return pd.DataFrame(

            metrics,

            index=[0],

        ).T.rename(

            columns={

                0: "Value"

            }

        )

    # ========================================================
    # RETURN TABLE
    # ========================================================

    def return_table(

        self,

        returns: pd.Series,

    ) -> pd.DataFrame:

        return pd.DataFrame({

            "Metric": [

                "Total Return",

                "Annual Return",

                "CAGR",

            ],

            "Value": [

                self.metrics.total_return(

                    returns

                ),

                self.metrics.annual_return(

                    returns

                ),

                self.metrics.cagr(

                    returns

                ),

            ],

        })

    def create(

        self,

        metrics: pd.Series | dict | None = None,

        trades: pd.DataFrame | None = None,

        equity_curve: pd.Series | pd.DataFrame | None = None,
    ) -> dict[str, Any]:

        if metrics is None:

            metrics = {}

        if isinstance(metrics, pd.Series):

            metrics = metrics.to_dict()

        return {

            "metrics": metrics,

            "trades": trades,

            "equity_curve": equity_curve,

        }

    def generate(

        self,

        metrics: pd.Series | dict | None = None,

        equity_curve: pd.Series | pd.DataFrame | None = None,
        trades: pd.DataFrame | None = None,
        portfolio: pd.DataFrame | None = None,
    ) -> dict[str, Any]:

        return self.create(

            metrics=metrics,

            trades=trades,

            equity_curve=equity_curve,

        )

    def save(self, report: dict[str, Any] | None, output: str | Path | None) -> str | None:

        if report is None:

            return None

        if output is None:

            return None

        output_path = Path(output)
        output_path.write_text("<html><body>Teardown report</body></html>", encoding="utf-8")
        return str(output_path)

    # ========================================================
    # RISK ADJUSTED TABLE
    # ========================================================

    def risk_adjusted_table(

        self,

        returns: pd.Series,

    ) -> pd.DataFrame:

        return pd.DataFrame({

            "Metric": [

                "Sharpe Ratio",

                "Sortino Ratio",

                "Calmar Ratio",

                "Omega Ratio",

            ],

            "Value": [

                self.metrics.sharpe_ratio(

                    returns

                ),

                self.metrics.sortino_ratio(

                    returns

                ),

                self.metrics.calmar_ratio(

                    returns

                ),

                self.metrics.omega_ratio(

                    returns

                ),

            ],

        })

    # ========================================================
    # TRADE SUMMARY
    # ========================================================

    def trade_summary(

        self,

        trades: pd.DataFrame,

    ) -> pd.DataFrame:

        if trades.empty:

            return pd.DataFrame()

        pnl = trades["PnL"]

        return pd.DataFrame({

            "Metric": [

                "Total Trades",

                "Winning Trades",

                "Losing Trades",

                "Average PnL",

                "Maximum Profit",

                "Maximum Loss",

            ],

            "Value": [

                len(trades),

                (pnl > 0).sum(),

                (pnl < 0).sum(),

                pnl.mean(),

                pnl.max(),

                pnl.min(),

            ],

        })

    # ========================================================
    # EQUITY CHART
    # ========================================================

    def equity_chart(

        self,

        returns: pd.Series,

        initial_capital: float = 100000,

    ):

        return self.plots.equity_curve(

            returns,

            initial_capital,

        )

    # ========================================================
    # CUMULATIVE RETURN CHART
    # ========================================================

    def cumulative_return_chart(

        self,

        returns: pd.Series,

    ):

        return self.plots.cumulative_returns(

            returns

        )

    # ========================================================
    # DRAWDOWN CHART
    # ========================================================

    def drawdown_chart(

        self,

        returns: pd.Series,

    ):

        return self.plots.drawdown(

            returns

        )

    # ========================================================
    # ROLLING VOLATILITY
    # ========================================================

    def rolling_volatility_chart(

        self,

        returns: pd.Series,

        window: int = 20,

    ):

        return self.plots.rolling_volatility(

            returns,

            window,

        )

    # ========================================================
    # ROLLING SHARPE
    # ========================================================

    def rolling_sharpe_chart(

        self,

        returns: pd.Series,

        window: int = 126,

    ):

        return self.plots.rolling_sharpe(

            returns,

            window,

            self.risk_free_rate,

            self.trading_days,

        )

    # ========================================================
    # MONTHLY RETURN HEATMAP
    # ========================================================

    def monthly_returns_chart(

        self,

        returns: pd.Series,

    ):

        return self.plots.monthly_returns_heatmap(

            returns

        )

    # ========================================================
    # RETURN DISTRIBUTION
    # ========================================================

    def return_distribution_chart(

        self,

        returns: pd.Series,

    ):

        return self.plots.return_distribution(

            returns

        )

    # ========================================================
    # BENCHMARK COMPARISON
    # ========================================================

    def benchmark_chart(

        self,

        portfolio_returns: pd.Series,

        benchmark_returns: pd.Series,

    ):

        return self.plots.benchmark_comparison(

            portfolio_returns,

            benchmark_returns,

        )

    # ========================================================
    # UNDERWATER PLOT
    # ========================================================

    def underwater_chart(

        self,

        returns: pd.Series,

    ):

        return self.plots.underwater_plot(

            returns

        )

    # ========================================================
    # ROLLING BETA
    # ========================================================

    def rolling_beta_chart(

        self,

        portfolio_returns: pd.Series,

        benchmark_returns: pd.Series,

        window: int = 126,

    ):

        return self.plots.rolling_beta(

            portfolio_returns,

            benchmark_returns,

            window,

        )

    # ========================================================
    # TRADE ANALYSIS
    # ========================================================

    def trade_analysis_chart(

        self,

        trades: pd.DataFrame,

    ):

        if trades.empty:

            return None

        return self.plots.trade_analysis(

            trades

        )

    # ========================================================
    # DASHBOARD
    # ========================================================

    def dashboard(

        self,

        returns: pd.Series,

        benchmark_returns: pd.Series | None = None,

        trades: pd.DataFrame | None = None,

        initial_capital: float = 100000,

    ) -> dict[str, Any]:

        dashboard = {

            "performance":

                self.performance_summary(

                    returns

                ),

            "risk":

                self.risk_table(

                    returns

                ),

            "returns":

                self.return_table(

                    returns

                ),

            "risk_adjusted":

                self.risk_adjusted_table(

                    returns

                ),

            "equity":

                self.equity_chart(

                    returns,

                    initial_capital,

                ),

            "drawdown":

                self.drawdown_chart(

                    returns

                ),

            "rolling_volatility":

                self.rolling_volatility_chart(

                    returns

                ),

            "rolling_sharpe":

                self.rolling_sharpe_chart(

                    returns

                ),

            "monthly":

                self.monthly_returns_chart(

                    returns

                ),

            "distribution":

                self.return_distribution_chart(

                    returns

                ),

        }

        if benchmark_returns is not None:

            dashboard["benchmark"] = (

                self.benchmark_summary(

                    returns,

                    benchmark_returns,

                )

            )

            dashboard["benchmark_chart"] = (

                self.benchmark_chart(

                    returns,

                    benchmark_returns,

                )

            )

            dashboard["rolling_beta"] = (

                self.rolling_beta_chart(

                    returns,

                    benchmark_returns,

                )

            )

        if trades is not None and not trades.empty:

            dashboard["trade_summary"] = (

                self.trade_summary(

                    trades

                )

            )

            dashboard["trade_chart"] = (

                self.trade_analysis_chart(

                    trades

                )

            )

        return dashboard

    # ========================================================
    # HTML REPORT
    # ========================================================

    def html_report(

        self,

        dashboard: dict[str, Any],

        filepath: str | Path,

    ):

        filepath = Path(filepath)

        filepath.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        html = [

            "<html>",

            "<head>",

            "<title>Portfolio Tear Sheet</title>",

            "</head>",

            "<body>",

            "<h1>Portfolio Tear Sheet</h1>",

        ]

        for key, value in dashboard.items():

            html.append(

                f"<h2>{key.replace('_',' ').title()}</h2>"

            )

            if isinstance(

                value,

                pd.DataFrame,

            ):

                html.append(

                    value.to_html()

                )

        html.extend(

            [

                "</body>",

                "</html>",

            ]

        )

        filepath.write_text(

            "\n".join(html),

            encoding="utf-8",

        )

    # ========================================================
    # PDF FIGURE EXPORT
    # ========================================================

    def export_figures(

        self,

        dashboard: dict[str, Any],

        directory: str | Path,

    ):

        directory = Path(directory)

        directory.mkdir(

            parents=True,

            exist_ok=True,

        )

        for name, obj in dashboard.items():

            if hasattr(

                obj,

                "savefig",

            ):

                obj.savefig(

                    directory / f"{name}.pdf",

                    bbox_inches="tight",

                )

    # ========================================================
    # REPORT SECTIONS
    # ========================================================

    def report_sections(

        self,

        dashboard: dict[str, Any],

    ) -> list[str]:

        return list(

            dashboard.keys()

        )

    # ========================================================
    # FIGURE COLLECTION
    # ========================================================

    def figures(

        self,

        dashboard: dict[str, Any],

    ) -> dict[str, Any]:

        figures = {}

        for key, value in dashboard.items():

            if hasattr(

                value,

                "savefig",

            ):

                figures[key] = value

        return figures

    # ========================================================
    # TABLE COLLECTION
    # ========================================================

    def tables(

        self,

        dashboard: dict[str, Any],

    ) -> dict[str, pd.DataFrame]:

        tables = {}

        for key, value in dashboard.items():

            if isinstance(

                value,

                pd.DataFrame,

            ):

                tables[key] = value

        return tables

    # ========================================================
    # COMPLETE REPORT
    # ========================================================

    def generate_report(

        self,

        returns: pd.Series,

        benchmark_returns: pd.Series | None = None,

        trades: pd.DataFrame | None = None,

        initial_capital: float = 100000,

    ) -> dict[str, Any]:

        return self.dashboard(

            returns=returns,

            benchmark_returns=benchmark_returns,

            trades=trades,

            initial_capital=initial_capital,

        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    def save_report(

        self,

        dashboard: dict[str, Any],

        output_directory: str | Path,

    ):

        output_directory = Path(

            output_directory

        )

        output_directory.mkdir(

            parents=True,

            exist_ok=True,

        )

        self.html_report(

            dashboard,

            output_directory / "report.html",

        )

        self.export_figures(

            dashboard,

            output_directory,

        )

    # ========================================================
    # REPORT METADATA
    # ========================================================

    def metadata(

        self,

    ) -> dict[str, Any]:

        return {

            "Trading Days": self.trading_days,

            "Risk Free Rate": self.risk_free_rate,

            "Engine": "Professional Tear Sheet",

            "Version": "1.0",

        }

    # ========================================================
    # SAVE TABLES
    # ========================================================

    def save_tables(

        self,

        dashboard: dict[str, Any],

        directory: str | Path,

    ):

        directory = Path(directory)

        directory.mkdir(

            parents=True,

            exist_ok=True,

        )

        for name, value in dashboard.items():

            if isinstance(

                value,

                pd.DataFrame,

            ):

                value.to_csv(

                    directory / f"{name}.csv",

                    index=True,

                )

    # ========================================================
    # EXPORT ALL
    # ========================================================

    def export(

        self,

        dashboard: dict[str, Any],

        directory: str | Path,

    ):

        directory = Path(directory)

        directory.mkdir(

            parents=True,

            exist_ok=True,

        )

        self.save_report(

            dashboard,

            directory,

        )

        self.save_tables(

            dashboard,

            directory,

        )

    # ========================================================
    # INFORMATION
    # ========================================================

    def info(

        self,

    ) -> dict[str, Any]:

        return {

            "Engine": "Professional Portfolio Tear Sheet",

            "Trading Days": self.trading_days,

            "Risk Free Rate": self.risk_free_rate,

            "Supported Outputs": [

                "Performance Summary",

                "Risk Summary",

                "Return Table",

                "Risk Adjusted Metrics",

                "Benchmark Summary",

                "Trade Summary",

                "Equity Curve",

                "Cumulative Returns",

                "Drawdown",

                "Rolling Volatility",

                "Rolling Sharpe",

                "Monthly Returns Heatmap",

                "Return Distribution",

                "Benchmark Comparison",

                "Rolling Beta",

                "Trade Analysis",

                "HTML Report",

                "CSV Tables",

                "PDF Figures",

            ],

        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(

        self,

    ):

        self.metrics = PerformanceMetrics(

            trading_days=252,

            risk_free_rate=0.06,

        )

        self.plots = PerformancePlots()

        self.trading_days = 252

        self.risk_free_rate = 0.06

    # ========================================================
    # UPDATE CONFIGURATION
    # ========================================================

    def configure(

        self,

        trading_days: int | None = None,

        risk_free_rate: float | None = None,

    ):

        if trading_days is not None:

            self.trading_days = trading_days

        if risk_free_rate is not None:

            self.risk_free_rate = risk_free_rate

        self.metrics = PerformanceMetrics(

            trading_days=self.trading_days,

            risk_free_rate=self.risk_free_rate,

        )

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(

        self,

    ):

        return (

            f"{self.__class__.__name__}("

            f"trading_days={self.trading_days}, "

            f"risk_free_rate={self.risk_free_rate:.2%})"

        )
