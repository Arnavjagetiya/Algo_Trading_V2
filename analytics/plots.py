"""
analytics/plots.py
=============================================================
Professional Performance Visualization Module
=============================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PerformancePlots:
    """
    Professional Backtesting Charts

    ✓ Equity Curve
    ✓ Cumulative Returns
    ✓ Daily Returns
    """

    def __init__(

        self,

        figsize: tuple[int, int] = (14, 7),

        dpi: int = 120,

        style: str = "ggplot",

    ):

        self.figsize = figsize

        self.dpi = dpi

        self.style = style

        plt.style.use(style)

    # =========================================================
    # FIGURE
    # =========================================================

    def _figure(

        self,

        title: str,

    ):

        fig, ax = plt.subplots(

            figsize=self.figsize,

            dpi=self.dpi,

        )

        ax.set_title(

            title,

            fontsize=15,

            fontweight="bold",

        )

        ax.grid(

            True,

            alpha=0.30,

        )

        return fig, ax

    # =========================================================
    # EQUITY CURVE
    # =========================================================

    def equity_curve(

        self,

        returns: pd.Series,

        initial_capital: float = 100000,

    ):

        equity = (

            1 + returns

        ).cumprod()

        equity *= initial_capital

        fig, ax = self._figure(

            "Equity Curve"

        )

        ax.plot(

            equity.index,

            equity.values,

            linewidth=2,

            label="Portfolio",

        )

        ax.set_ylabel(

            "Portfolio Value"

        )

        ax.legend()

        return fig

    # =========================================================
    # CUMULATIVE RETURNS
    # =========================================================

    def cumulative_returns(

        self,

        returns: pd.Series,

    ):

        cumulative = (

            1 + returns

        ).cumprod() - 1

        fig, ax = self._figure(

            "Cumulative Returns"

        )

        ax.plot(

            cumulative.index,

            cumulative.values * 100,

            linewidth=2,

        )

        ax.set_ylabel(

            "Return (%)"

        )

        return fig

    # =========================================================
    # DAILY RETURNS
    # =========================================================

    def daily_returns(

        self,

        returns: pd.Series,

    ):

        fig, ax = self._figure(

            "Daily Returns"

        )

        ax.bar(

            returns.index,

            returns.values * 100,

            width=1,

        )

        ax.axhline(

            0,

            linewidth=1,

        )

        ax.set_ylabel(

            "Daily Return (%)"

        )

        return fig

    # =========================================================
    # HISTOGRAM
    # =========================================================

    def returns_histogram(

        self,

        returns: pd.Series,

        bins: int = 50,

    ):

        fig, ax = self._figure(

            "Return Distribution"

        )

        ax.hist(

            returns,

            bins=bins,

        )

        ax.set_xlabel(

            "Return"

        )

        ax.set_ylabel(

            "Frequency"

        )

        return fig

    # =========================================================
    # CUMULATIVE GROWTH
    # =========================================================

    def growth_of_one(

        self,

        returns: pd.Series,

    ):

        growth = (

            1 + returns

        ).cumprod()

        fig, ax = self._figure(

            "Growth of ₹1"

        )

        ax.plot(

            growth.index,

            growth.values,

            linewidth=2,

        )

        ax.set_ylabel(

            "Portfolio Growth"

        )

        return fig

    # =========================================================
    # DRAWDOWN
    # =========================================================

    def drawdown(
        self,
        returns: pd.Series,
    ):

        equity = (
            1 + returns
        ).cumprod()

        peak = equity.cummax()

        drawdown = (
            equity - peak
        ) / peak

        fig, ax = self._figure(
            "Drawdown"
        )

        ax.fill_between(
            drawdown.index,
            drawdown.values * 100,
            0,
            alpha=0.35,
        )

        ax.set_ylabel(
            "Drawdown (%)"
        )

        return fig

    # =========================================================
    # ROLLING VOLATILITY
    # =========================================================

    def rolling_volatility(
        self,
        returns: pd.Series,
        window: int = 20,
        trading_days: int = 252,
    ):

        volatility = (

            returns

            .rolling(window)

            .std()

            * np.sqrt(trading_days)

            * 100

        )

        fig, ax = self._figure(
            "Rolling Volatility"
        )

        ax.plot(
            volatility.index,
            volatility.values,
            linewidth=2,
        )

        ax.set_ylabel(
            "Volatility (%)"
        )

        return fig

    # =========================================================
    # ROLLING SHARPE
    # =========================================================

    def rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 126,
        risk_free_rate: float = 0.06,
        trading_days: int = 252,
    ):

        def sharpe(x):

            excess = (
                x
                - risk_free_rate / trading_days
            )

            std = excess.std()

            if std == 0:

                return np.nan

            return (

                excess.mean()

                / std

                * np.sqrt(trading_days)

            )

        values = (

            returns

            .rolling(window)

            .apply(

                sharpe,

                raw=False,

            )

        )

        fig, ax = self._figure(
            "Rolling Sharpe Ratio"
        )

        ax.plot(
            values.index,
            values.values,
            linewidth=2,
        )

        ax.axhline(
            1,
            linestyle="--",
        )

        ax.set_ylabel(
            "Sharpe Ratio"
        )

        return fig

    # =========================================================
    # MONTHLY RETURNS HEATMAP
    # =========================================================

    def monthly_returns_heatmap(
        self,
        returns: pd.Series,
    ):

        monthly = (

            returns

            .resample("M")

            .apply(

                lambda x: (1 + x).prod() - 1

            )

        )

        table = monthly.to_frame(
            "Return"
        )

        table["Year"] = table.index.year

        table["Month"] = table.index.strftime(
            "%b"
        )

        pivot = table.pivot(

            index="Year",

            columns="Month",

            values="Return",

        )

        fig, ax = self._figure(
            "Monthly Returns Heatmap"
        )

        image = ax.imshow(

            pivot.fillna(0),

            aspect="auto",

        )

        ax.set_xticks(
            range(len(pivot.columns))
        )

        ax.set_xticklabels(
            pivot.columns
        )

        ax.set_yticks(
            range(len(pivot.index))
        )

        ax.set_yticklabels(
            pivot.index
        )

        plt.colorbar(
            image,
            ax=ax,
        )

        return fig

    # =========================================================
    # RETURN DISTRIBUTION
    # =========================================================

    def return_distribution(
        self,
        returns: pd.Series,
        bins: int = 60,
    ):

        fig, ax = self._figure(
            "Return Distribution"
        )

        ax.hist(
            returns,
            bins=bins,
        )

        ax.axvline(
            returns.mean(),
            linewidth=2,
        )

        ax.set_xlabel(
            "Returns"
        )

        ax.set_ylabel(
            "Frequency"
        )

        return fig

    # =========================================================
    # ROLLING MEAN RETURN
    # =========================================================

    def rolling_return(
        self,
        returns: pd.Series,
        window: int = 20,
    ):

        rolling = (

            returns

            .rolling(window)

            .mean()

            * 100

        )

        fig, ax = self._figure(
            "Rolling Mean Return"
        )

        ax.plot(
            rolling.index,
            rolling.values,
            linewidth=2,
        )

        ax.set_ylabel(
            "Return (%)"
        )

        return fig

    # =========================================================
    # BENCHMARK COMPARISON
    # =========================================================

    def benchmark_comparison(

        self,

        portfolio_returns: pd.Series,

        benchmark_returns: pd.Series,

    ):

        portfolio = (

            1 + portfolio_returns

        ).cumprod()

        benchmark = (

            1 + benchmark_returns

        ).cumprod()

        fig, ax = self._figure(

            "Portfolio vs Benchmark"

        )

        ax.plot(

            portfolio.index,

            portfolio.values,

            linewidth=2,

            label="Portfolio",

        )

        ax.plot(

            benchmark.index,

            benchmark.values,

            linewidth=2,

            label="Benchmark",

        )

        ax.set_ylabel(

            "Growth"

        )

        ax.legend()

        return fig

    # =========================================================
    # PORTFOLIO ALLOCATION
    # =========================================================

    def portfolio_allocation(

        self,

        weights: pd.Series,

    ):

        fig, ax = self._figure(

            "Portfolio Allocation"

        )

        ax.pie(

            weights.values,

            labels=weights.index,

            autopct="%1.1f%%",

            startangle=90,

        )

        ax.axis(

            "equal"

        )

        return fig

    # =========================================================
    # TRADE ANALYSIS
    # =========================================================

    def trade_analysis(

        self,

        trades: pd.DataFrame,

    ):

        fig, ax = self._figure(

            "Trade P&L"

        )

        pnl = trades["PnL"]

        colors = np.where(

            pnl >= 0,

            "green",

            "red",

        )

        ax.bar(

            range(len(pnl)),

            pnl.values,

        )

        for bar, color in zip(

            ax.patches,

            colors,

        ):

            bar.set_color(

                color

            )

        ax.set_xlabel(

            "Trade"

        )

        ax.set_ylabel(

            "Profit / Loss"

        )

        return fig

    # =========================================================
    # UNDERWATER PLOT
    # =========================================================

    def underwater_plot(

        self,

        returns: pd.Series,

    ):

        equity = (

            1 + returns

        ).cumprod()

        peaks = equity.cummax()

        drawdown = (

            equity

            - peaks

        ) / peaks

        fig, ax = self._figure(

            "Underwater Plot"

        )

        ax.fill_between(

            drawdown.index,

            drawdown.values * 100,

            0,

            alpha=0.35,

        )

        ax.set_ylabel(

            "Drawdown (%)"

        )

        return fig

    # =========================================================
    # ROLLING BETA
    # =========================================================

    def rolling_beta(

        self,

        portfolio_returns: pd.Series,

        benchmark_returns: pd.Series,

        window: int = 126,

    ):

        data = pd.concat(

            [

                portfolio_returns,

                benchmark_returns,

            ],

            axis=1,

        ).dropna()

        beta = pd.Series(

            index=data.index,

            dtype=float,

        )

        for i in range(

            window,

            len(data),

        ):

            subset = data.iloc[

                i - window:i

            ]

            covariance = np.cov(

                subset.iloc[:, 0],

                subset.iloc[:, 1],

            )[0, 1]

            variance = np.var(

                subset.iloc[:, 1]

            )

            if variance == 0:

                value = np.nan

            else:

                value = (

                    covariance

                    / variance

                )

            beta.iloc[i] = value

        fig, ax = self._figure(

            "Rolling Beta"

        )

        ax.plot(

            beta.index,

            beta.values,

            linewidth=2,

        )

        ax.axhline(

            1,

            linestyle="--",

        )

        ax.set_ylabel(

            "Beta"

        )

        return fig

    # =========================================================
    # EXPOSURE
    # =========================================================

    def exposure(

        self,

        exposure: pd.Series,

    ):

        fig, ax = self._figure(

            "Portfolio Exposure"

        )

        ax.plot(

            exposure.index,

            exposure.values * 100,

            linewidth=2,

        )

        ax.set_ylabel(

            "Exposure (%)"

        )

        return fig

    # =========================================================
    # DASHBOARD
    # =========================================================

    def dashboard(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
        initial_capital: float = 100000,
    ):

        figures = {

            "equity_curve": self.equity_curve(
                returns,
                initial_capital,
            ),

            "cumulative_returns": self.cumulative_returns(
                returns,
            ),

            "drawdown": self.drawdown(
                returns,
            ),

            "rolling_volatility": self.rolling_volatility(
                returns,
            ),

            "rolling_sharpe": self.rolling_sharpe(
                returns,
            ),

        }

        if benchmark_returns is not None:

            figures["benchmark"] = self.benchmark_comparison(

                returns,

                benchmark_returns,

            )

        return figures

    # =========================================================
    # SAVE FIGURE
    # =========================================================

    @staticmethod
    def save(
        figure,
        filepath: str | Path,
        dpi: int = 300,
        bbox_inches: str = "tight",
    ):

        filepath = Path(filepath)

        filepath.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        figure.savefig(

            filepath,

            dpi=dpi,

            bbox_inches=bbox_inches,

        )

    # =========================================================
    # SAVE ALL
    # =========================================================

    def save_all(
        self,
        figures: dict[str, plt.Figure],
        directory: str | Path,
    ):

        directory = Path(directory)

        directory.mkdir(

            parents=True,

            exist_ok=True,

        )

        for name, fig in figures.items():

            self.save(

                fig,

                directory / f"{name}.png",

            )

    # =========================================================
    # SHOW
    # =========================================================

    @staticmethod
    def show():

        plt.show()

    # =========================================================
    # CLOSE
    # =========================================================

    @staticmethod
    def close(
        figure=None,
    ):

        if figure is None:

            plt.close("all")

        else:

            plt.close(figure)

    # =========================================================
    # EXPORT PDF REPORT (FIGURES ONLY)
    # =========================================================

    def export_report(
        self,
        figures: dict[str, plt.Figure],
        directory: str | Path,
    ):

        directory = Path(directory)

        directory.mkdir(

            parents=True,

            exist_ok=True,

        )

        for name, figure in figures.items():

            figure.savefig(

                directory / f"{name}.pdf",

                dpi=self.dpi,

                bbox_inches="tight",

            )

    # =========================================================
    # INFORMATION
    # =========================================================

    def info(
        self,
    ) -> dict:

        return {

            "Style": self.style,

            "Figure Size": self.figsize,

            "DPI": self.dpi,

            "Available Charts": [

                "Equity Curve",

                "Cumulative Returns",

                "Daily Returns",

                "Returns Histogram",

                "Growth of ₹1",

                "Drawdown",

                "Rolling Volatility",

                "Rolling Sharpe",

                "Monthly Returns Heatmap",

                "Return Distribution",

                "Rolling Mean Return",

                "Portfolio vs Benchmark",

                "Portfolio Allocation",

                "Trade Analysis",

                "Underwater Plot",

                "Rolling Beta",

                "Portfolio Exposure",

                "Dashboard",

            ],

        }

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
    ):

        self.figsize = (14, 7)

        self.dpi = 120

        self.style = "ggplot"

        plt.style.use(self.style)

    # =========================================================
    # COMPATIBILITY METHODS
    # =========================================================

    def generate_all(
        self,
        equity_curve: pd.Series | pd.DataFrame | None = None,
        trades: pd.DataFrame | None = None,
        portfolio: pd.DataFrame | None = None,
    ):

        if equity_curve is None:

            return None

        if isinstance(equity_curve, pd.DataFrame):

            returns = equity_curve.iloc[:, 0] if equity_curve.shape[1] > 0 else pd.Series(dtype=float)

        else:

            returns = equity_curve

        if isinstance(returns, pd.Series):

            self.equity_curve(returns)
            self.cumulative_returns(returns)
            self.daily_returns(returns)
            self.return_distribution(returns)

        return True

    def plot_all(
        self,
        equity_curve: pd.Series | pd.DataFrame | None = None,
        trades: pd.DataFrame | None = None,
    ):

        return self.generate_all(
            equity_curve=equity_curve,
            trades=trades,
            portfolio=None,
        )

    # =========================================================
    # REPRESENTATION
    # =========================================================

    def __repr__(
        self,
    ):

        return (

            f"{self.__class__.__name__}("

            f"figsize={self.figsize}, "

            f"dpi={self.dpi}, "

            f"style='{self.style}')"

        )