"""
backtest_engine.py
=============================================================
Professional Event-Driven Portfolio Backtesting Engine

Author : Arnav
Python : 3.12+
=============================================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG

# =============================================================
# DATA
# =============================================================

from data.universe import Universe
from data.downloader import Downloader
from data.preprocess import Preprocessor

# =============================================================
# INDICATORS
# =============================================================

from indicators.technicals import TechnicalIndicators
from indicators.factors import FactorEngine
from indicators.volatility import VolatilityIndicators

# =============================================================
# MODELS
# =============================================================

from models.clustering import StockClusterer
from models.garch import GARCHModel
from models.optimizer import PortfolioOptimizer

# =============================================================
# STRATEGY
# =============================================================

from strategy.signal import SignalEngine
from strategy.risk import RiskManager
from strategy.portfolio import Portfolio
from strategy.execution import ExecutionEngine

# =============================================================
# ANALYTICS
# =============================================================

from analytics.metrics import PerformanceMetrics
from analytics.plots import PerformancePlots
from analytics.tearsheet import TearSheet

# =============================================================
# ENGINE STATE
# =============================================================


@dataclass(slots=True)
class EngineState:

    start_time: float = 0.0

    end_time: float = 0.0

    running: bool = False

    completed: bool = False

    current_symbol: str = ""

    processed_symbols: int = 0

    total_symbols: int = 0

    errors: list[str] = field(default_factory=list)


# =============================================================
# BACKTEST ENGINE
# =============================================================


class BacktestEngine:

    """
    Professional Event-Driven Portfolio Backtester
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(

        self,

        config=CONFIG,

    ):

        self.config = config

        self.state = EngineState()

        self.logger = self._build_logger()

        self._create_directories()

        self._initialize_modules()

        self._initialize_runtime()

    # =========================================================
    # LOGGER
    # =========================================================

    def _build_logger(self):

        logger = logging.getLogger(

            "BacktestEngine"

        )

        logger.setLevel(logging.INFO)

        if logger.handlers:

            return logger

        formatter = logging.Formatter(

            "%(asctime)s | %(levelname)s | %(message)s"

        )

        console = logging.StreamHandler()

        console.setFormatter(formatter)

        logger.addHandler(console)

        return logger

    # =========================================================
    # DIRECTORIES
    # =========================================================

    def _create_directories(self):

        directories = [

            self.config.data_directory,

            self.config.cache_directory,

            self.config.log_directory,

            self.config.output_directory,

        ]

        for directory in directories:

            Path(directory).mkdir(

                parents=True,

                exist_ok=True,

            )

    # =========================================================
    # MODULE INITIALIZATION
    # =========================================================

    def _initialize_modules(self):

        self.logger.info(

            "Initializing framework modules..."

        )

        # ----------------------------
        # DATA
        # ----------------------------

        self.universe = Universe()

        self.downloader = Downloader()

        self.preprocessor = Preprocessor()

        # ----------------------------
        # INDICATORS
        # ----------------------------

        self.technicals = TechnicalIndicators()

        self.factors = FactorEngine()

        self.volatility = VolatilityIndicators()

        # ----------------------------
        # MODELS
        # ----------------------------

        self.clustering = StockClusterer()

        self.garch = GARCHModel()

        self.optimizer = PortfolioOptimizer()

        # ----------------------------
        # STRATEGY
        # ----------------------------

        self.signal_engine = SignalEngine()

        self.risk_manager = RiskManager()

        self.portfolio = Portfolio()

        self.execution = ExecutionEngine()

        # ----------------------------
        # ANALYTICS
        # ----------------------------

        self.metrics = PerformanceMetrics(

            risk_free_rate=CONFIG.risk_free_rate

            if hasattr(CONFIG, "risk_free_rate")

            else 0.06

        )

        self.plots = PerformancePlots()

        self.tearsheet = TearSheet()

    # =========================================================
    # RUNTIME VARIABLES
    # =========================================================

    def _initialize_runtime(self):

        self.market_data: dict[str, pd.DataFrame] = {}

        self.cleaned_data: dict[str, pd.DataFrame] = {}

        self.indicator_data: dict[str, pd.DataFrame] = {}

        self.factor_data: dict[str, pd.DataFrame] = {}

        self.signal_data: dict[str, pd.DataFrame] = {}

        self.portfolio_history = []

        self.trade_history = []

        self.position_history = []

        self.equity_curve = []

        self.performance = {}

        self.report = None

    # =========================================================
    # START
    # =========================================================

    def start(self):

        self.state.running = True

        self.state.completed = False

        self.state.start_time = time.time()

        self.logger.info(

            "=" * 60

        )

        self.logger.info(

            "Starting Backtest"

        )

        self.logger.info(

            "=" * 60

        )

    # =========================================================
    # STOP
    # =========================================================

    def stop(self):

        self.state.running = False

        self.state.completed = True

        self.state.end_time = time.time()

        elapsed = (

            self.state.end_time

            - self.state.start_time

        )

        self.logger.info(

            f"Finished in {elapsed:.2f} seconds."

        )

    # =========================================================
    # LOAD UNIVERSE
    # =========================================================

    def load_universe(self):

        self.logger.info(

            "Loading trading universe..."

        )

        symbols = self.universe.NIFTY50

        if not symbols:

            raise RuntimeError(

                "Universe is empty."

            )

        self.state.total_symbols = len(

            symbols

        )

        self.logger.info(

            f"{len(symbols)} symbols loaded."

        )

        return symbols

    # =========================================================
    # DOWNLOAD DATA
    # =========================================================

    def download_market_data(

        self,

        symbols: list[str],

    ):

        self.logger.info(

            "Downloading historical market data..."

        )

        data = self.downloader.download(

            symbols=symbols,

            start=self.config.start_date,

            end=self.config.end_date,

            interval=self.config.interval,

            use_cache=getattr(

                self.config,

                "cache_enabled",

                False,

            ),

        )

        if not data:

            raise RuntimeError(

                "Downloader returned no data."

            )

        self.market_data = {

            symbol: df

            for symbol, df in data.items()

            if df is not None and not getattr(df, "empty", False)

        }

        self.logger.info(

            f"Downloaded {len(self.market_data)} datasets."

        )

        return self.market_data

    # =========================================================
    # CACHE DATA
    # =========================================================

    def cache_market_data(self):

        if not getattr(

            self.config,

            "cache_enabled",

            False,

        ):

            return

        cache_dir = Path(

            self.config.cache_directory

        )

        for symbol, df in self.market_data.items():

            try:

                file = cache_dir / f"{symbol}.parquet"

                df.to_parquet(

                    file,

                    index=True,

                )

            except Exception as exc:

                self.logger.warning(

                    f"Cache failed for {symbol}: {exc}"

                )

    # =========================================================
    # LOAD CACHE
    # =========================================================

    def load_cached_data(

        self,

        symbols: list[str],

    ):

        cache = {}

        cache_dir = Path(

            self.config.cache_directory

        )

        for symbol in symbols:

            file = cache_dir / f"{symbol}.parquet"

            if file.exists():

                try:

                    cache[symbol] = pd.read_parquet(

                        file

                    )

                except Exception:

                    pass

        if cache:

            self.logger.info(

                f"Loaded {len(cache)} cached datasets."

            )

        return cache

    # =========================================================
    # VALIDATE DATA
    # =========================================================

    def validate_market_data(self):

        self.logger.info(

            "Validating downloaded data..."

        )

        valid = {}

        for symbol, df in self.market_data.items():

            if df is None:

                continue

            if df.empty:

                continue

            required = {

                "Open",

                "High",

                "Low",

                "Close",

                "Volume",

            }

            if not required.issubset(

                df.columns

            ):

                self.logger.warning(

                    f"{symbol} missing OHLCV columns."

                )

                continue

            if len(df) < 50:

                self.logger.warning(

                    f"{symbol} ignored (insufficient history)."

                )

                continue

            valid[symbol] = df.copy()

        self.market_data = valid

        self.logger.info(

            f"{len(valid)} datasets validated."

        )

    # =========================================================
    # PREPROCESS DATA
    # =========================================================

    def preprocess_data(self):

        self.logger.info(

            "Running preprocessing..."

        )

        processed = {}

        for symbol, df in self.market_data.items():

            try:

                clean = self.preprocessor.process(

                    df.copy()

                )

                processed[symbol] = clean

            except Exception as exc:

                self.state.errors.append(

                    f"{symbol}: {exc}"

                )

                self.logger.exception(

                    f"Preprocessing failed: {symbol}"

                )

        self.cleaned_data = processed

        self.logger.info(

            f"Processed {len(processed)} symbols."

        )

    # =========================================================
    # DATA SUMMARY
    # =========================================================

    def data_summary(self):

        summary = []

        for symbol, df in self.cleaned_data.items():

            summary.append(

                {

                    "Symbol": symbol,

                    "Rows": len(df),

                    "Start": df.index.min(),

                    "End": df.index.max(),

                    "Missing": int(

                        df.isna().sum().sum()

                    ),

                }

            )

        return pd.DataFrame(

            summary

        )

    # =========================================================
    # PREPARE DATA
    # =========================================================

    def prepare_data(self):

        symbols = self.load_universe()

        cached = {}

        if getattr(

            self.config,

            "cache_enabled",

            False,

        ):

            cached = self.load_cached_data(

                symbols

            )

        missing = [

            s

            for s in symbols

            if s not in cached

        ]

        self.market_data = cached

        if missing:

            downloaded = self.download_market_data(

                missing

            )

            self.market_data.update(

                downloaded

            )

            self.cache_market_data()

        self.validate_market_data()

        self.preprocess_data()

        self.logger.info(

            "Market data preparation completed."

        )

    # =========================================================
    # CALCULATE INDICATORS
    # =========================================================

    def calculate_indicators(self):

        self.logger.info(

            "Calculating technical indicators..."

        )

        indicator_data = {}

        for symbol, df in self.cleaned_data.items():

            try:

                data = df.copy()

                data = self.technicals.calculate_all(

                    data

                )

                indicator_data[symbol] = data

                self.state.processed_symbols += 1

            except Exception as exc:

                self.logger.exception(

                    f"Indicator calculation failed: {symbol}"

                )

                self.state.errors.append(

                    f"{symbol}: {exc}"

                )

        self.indicator_data = indicator_data

        self.logger.info(

            f"Indicators calculated for {len(indicator_data)} symbols."

        )

    # =========================================================
    # FACTOR GENERATION
    # =========================================================

    def generate_factors(self):

        self.logger.info(

            "Generating factor models..."

        )

        factor_data = {}

        for symbol, df in self.indicator_data.items():

            try:

                factors = self.factors.calculate_all(

                    df.copy()

                )

                factor_data[symbol] = factors

            except Exception as exc:

                self.logger.exception(

                    f"Factor generation failed: {symbol}"

                )

                self.state.errors.append(

                    f"{symbol}: {exc}"

                )

        self.factor_data = factor_data

        self.logger.info(

            f"Factors generated for {len(factor_data)} symbols."

        )

    # =========================================================
    # VOLATILITY
    # =========================================================

    def estimate_volatility(self):

        self.logger.info(

            "Estimating volatility..."

        )

        for symbol, df in self.factor_data.items():

            try:

                returns = (

                    df["Close"]

                    .pct_change()

                    .dropna()

                )

                df["HistoricalVolatility"] = (

                    self.volatility

                    .historical_volatility(

                        returns

                    )

                )

                df["EWMAVolatility"] = (

                    self.volatility

                    .ewma_volatility(

                        returns

                    )

                )

            except Exception as exc:

                self.logger.exception(

                    f"Volatility failed: {symbol}"

                )

                self.state.errors.append(

                    f"{symbol}: {exc}"

                )

    # =========================================================
    # GARCH
    # =========================================================

    def fit_garch_models(self):

        self.logger.info(

            "Running GARCH models..."

        )

        for symbol, df in self.factor_data.items():

            try:

                returns = (

                    df["Close"]

                    .pct_change()

                    .dropna()

                )

                result = self.garch.fit(

                    returns

                )

                forecast = self.garch.forecast(

                    result

                )

                df["GARCHVolatility"] = (

                    forecast

                )

            except Exception as exc:

                self.logger.warning(

                    f"GARCH skipped for {symbol}: {exc}"

                )

    # =========================================================
    # CLUSTERING
    # =========================================================

    def cluster_universe(self):

        self.logger.info(

            "Clustering universe..."

        )

        features = []

        symbols = []

        for symbol, df in self.factor_data.items():

            try:

                latest = df.iloc[-1]

                vector = latest.select_dtypes(

                    include=np.number

                )

                features.append(

                    vector.values

                )

                symbols.append(

                    symbol

                )

            except Exception:

                pass

        if len(features) < 2:

            self.logger.warning(

                "Not enough securities for clustering."

            )

            return

        feature_matrix = np.vstack(

            features

        )

        labels = self.clustering.fit_predict(

            feature_matrix

        )

        self.cluster_labels = dict(

            zip(

                symbols,

                labels,

            )

        )

        self.logger.info(

            f"{len(set(labels))} clusters created."

        )

    # =========================================================
    # PORTFOLIO OPTIMIZATION
    # =========================================================

    def optimize_portfolio(self):

        self.logger.info(

            "Optimizing portfolio weights..."

        )

        returns = pd.DataFrame({

            symbol:

            df["Close"].pct_change()

            for symbol, df in self.factor_data.items()

        }).dropna()

        if returns.empty:

            self.optimized_weights = {}

            return

        try:

            weights = self.optimizer.optimize(

                returns

            )

            self.optimized_weights = weights

        except Exception as exc:

            self.logger.warning(

                f"Optimization failed: {exc}"

            )

            equal = 1.0 / len(

                returns.columns

            )

            self.optimized_weights = {

                s: equal

                for s in returns.columns

            }

    # =========================================================
    # FEATURE PIPELINE
    # =========================================================

    def prepare_features(self):

        self.calculate_indicators()

        self.generate_factors()

        self.estimate_volatility()

        self.fit_garch_models()

        self.cluster_universe()

        self.optimize_portfolio()

        self.logger.info(

            "Feature engineering completed."

        )

    # =========================================================
    # GENERATE SIGNALS
    # =========================================================

    def generate_signals(self):

        self.logger.info(

            "Generating trading signals..."

        )

        signal_data = {}

        for symbol, df in self.factor_data.items():

            try:

                signals = self.signal_engine.generate(

                    df.copy()

                )

                signal_data[symbol] = signals

            except Exception as exc:

                self.logger.exception(

                    f"Signal generation failed: {symbol}"

                )

                self.state.errors.append(

                    f"{symbol}: {exc}"

                )

        self.signal_data = signal_data

        self.logger.info(

            f"Signals generated for {len(signal_data)} symbols."

        )

    # =========================================================
    # BUILD PORTFOLIO
    # =========================================================

    def build_portfolio(self):

        self.logger.info(

            "Constructing portfolio..."

        )

        try:

            self.portfolio.initialize(

                capital=self.config.initial_capital,

            )

        except AttributeError:

            self.portfolio.initialize(

                self.config.initial_capital

            )

        for symbol, weight in self.optimized_weights.items():

            try:

                self.portfolio.set_target_weight(

                    symbol,

                    weight,

                )

            except AttributeError:

                pass

    # =========================================================
    # POSITION SIZING
    # =========================================================

    def calculate_position_sizes(self):

        self.logger.info(

            "Calculating position sizes..."

        )

        self.position_sizes = {}

        capital = self.config.initial_capital

        risk = self.config.risk_per_trade

        for symbol, df in self.signal_data.items():

            try:

                source = self.cleaned_data.get(symbol)

                if source is None:

                    raise KeyError(f"No source data for {symbol}")

                if "Close" in df.columns:

                    price_series = df["Close"]

                else:

                    price_series = source["Close"]

                if price_series is None or len(price_series) == 0:

                    continue

                price = float(price_series.dropna().iloc[-1])

                if price <= 0:

                    continue

                allocation = (

                    capital

                    * risk

                )

                quantity = max(

                    1,

                    int(

                        allocation

                        / price

                    ),

                )

                self.position_sizes[symbol] = quantity

            except Exception as exc:

                self.logger.warning(

                    f"Position sizing failed for {symbol}: {exc}"

                )

    # =========================================================
    # APPLY RISK MANAGEMENT
    # =========================================================

    def apply_risk_management(self):

        self.logger.info(

            "Applying risk management..."

        )

        filtered = {}

        for symbol, df in self.signal_data.items():

            try:

                approved = self.risk_manager.evaluate(

                    data=df,

                    symbol=symbol,

                )

                if approved:

                    filtered[symbol] = df

            except AttributeError:

                try:

                    approved = self.risk_manager.validate(

                        df

                    )

                    if approved:

                        filtered[symbol] = df

                except Exception:

                    filtered[symbol] = df

            except Exception as exc:

                self.logger.warning(

                    f"Risk evaluation failed for {symbol}: {exc}"

                )

        self.signal_data = filtered

    # =========================================================
    # FILTER TRADES
    # =========================================================

    def filter_trade_candidates(self):

        self.logger.info(

            "Selecting executable trades..."

        )

        candidates = {}

        for symbol, df in self.signal_data.items():

            try:

                latest = df.iloc[-1]

                signal = latest.get(

                    "Signal",

                    0,

                )

                if signal != 0:

                    candidates[symbol] = latest

            except Exception:

                continue

        self.trade_candidates = candidates

        self.logger.info(

            f"{len(candidates)} trade candidates selected."

        )

    # =========================================================
    # PREPARE EXECUTION QUEUE
    # =========================================================

    def prepare_execution_queue(self):

        queue = []

        for symbol, row in self.trade_candidates.items():

            if isinstance(row, pd.Series):

                price = row.get("Close")

                if price is None:

                    price = row.get("Price")

                if price is None:

                    try:

                        price = self.cleaned_data[symbol]["Close"].iloc[-1]

                    except Exception:

                        price = 0.0

            else:

                price = 0.0

            queue.append(

                {

                    "symbol": symbol,

                    "signal": row.get(

                        "Signal",

                        0,

                    ) if isinstance(row, pd.Series) else 0,

                    "price": price,

                    "quantity": self.position_sizes.get(

                        symbol,

                        0,

                    ),

                }

            )

        self.execution_queue = queue

        self.logger.info(

            f"{len(queue)} orders queued."

        )

    # =========================================================
    # STRATEGY PIPELINE
    # =========================================================

    def prepare_strategy(self):

        self.generate_signals()

        self.build_portfolio()

        self.calculate_position_sizes()

        self.apply_risk_management()

        self.filter_trade_candidates()

        self.prepare_execution_queue()

        self.logger.info(

            "Strategy preparation completed."

        )
        
    # =========================================================
    # EVENT-DRIVEN BACKTEST LOOP
    # =========================================================

    def run_backtest(self):

        self.logger.info(

            "Starting event-driven backtest..."

        )

        self.trade_history.clear()

        self.position_history.clear()

        self.portfolio_history.clear()

        self.equity_curve.clear()

        symbols = list(

            self.signal_data.keys()

        )

        if not symbols:

            self.logger.warning(

                "No symbols available."

            )

            return

        common_dates = None

        for symbol in symbols:

            dates = self.signal_data[symbol].index

            if common_dates is None:

                common_dates = dates

            else:

                common_dates = common_dates.intersection(

                    dates

                )

        common_dates = common_dates.sort_values()

        for current_date in common_dates:

            self.process_bar(

                current_date

            )

        self.logger.info(

            "Backtest completed."

        )

    # =========================================================
    # PROCESS SINGLE BAR
    # =========================================================

    def process_bar(

        self,

        current_date,

    ):

        self.manage_existing_positions(

            current_date

        )

        self.execute_pending_orders(

            current_date

        )

        self.update_portfolio_value(

            current_date

        )

        self.record_daily_snapshot(

            current_date

        )

    # =========================================================
    # POSITION MANAGEMENT
    # =========================================================

    def manage_existing_positions(

        self,

        current_date,

    ):

        try:

            self.portfolio.update_positions(

                current_date

            )

        except AttributeError:

            pass

        try:

            self.risk_manager.manage_positions(

                self.portfolio,

                current_date,

            )

        except AttributeError:

            pass

    # =========================================================
    # EXECUTE ORDERS
    # =========================================================

    def execute_pending_orders(

        self,

        current_date,

    ):

        for order in self.execution_queue:

            symbol = order["symbol"]

            quantity = order["quantity"]

            signal = order["signal"]

            df = self.signal_data[symbol]

            if current_date not in df.index:

                continue

            row = df.loc[current_date]

            if signal == 0:

                continue

            price = row.get("Close")

            if price is None:

                price = row.get("Price")

            if price is None:

                try:

                    price = self.cleaned_data[symbol]["Close"].iloc[-1]

                except Exception:

                    price = 0.0

            try:

                fill = self.execution.execute(

                    symbol=symbol,

                    signal=signal,

                    quantity=quantity,

                    price=price,

                    timestamp=current_date,

                )

            except AttributeError:

                fill = {

                    "symbol": symbol,

                    "quantity": quantity,

                    "signal": signal,

                    "price": price,

                    "timestamp": current_date,

                }

            self.trade_history.append(

                fill

            )

            try:

                self.portfolio.process_fill(

                    fill

                )

            except AttributeError:

                pass

    # =========================================================
    # UPDATE PORTFOLIO VALUE
    # =========================================================

    def update_portfolio_value(

        self,

        current_date,

    ):

        try:

            value = self.portfolio.market_value(

                self.signal_data,

                current_date,

            )

        except AttributeError:

            value = self.config.initial_capital

            for trade in self.trade_history:

                value += 0.0

        self.equity_curve.append(

            {

                "Date": current_date,

                "Equity": value,

            }

        )

    # =========================================================
    # RECORD DAILY STATE
    # =========================================================

    def record_daily_snapshot(

        self,

        current_date,

    ):

        snapshot = {

            "Date": current_date,

            "Cash": getattr(

                self.portfolio,

                "cash",

                np.nan,

            ),

            "Equity": self.equity_curve[-1][

                "Equity"

            ],

            "Trades": len(

                self.trade_history

            ),

        }

        self.portfolio_history.append(

            snapshot

        )

        try:

            positions = self.portfolio.positions

            self.position_history.append(

                {

                    "Date": current_date,

                    "Positions": positions.copy(),

                }

            )

        except AttributeError:

            pass

    # =========================================================
    # CONVERT TO DATAFRAMES
    # =========================================================

    def finalize_backtest_data(self):

        self.trade_history = pd.DataFrame(

            self.trade_history

        )

        self.portfolio_history = pd.DataFrame(

            self.portfolio_history

        )

        self.position_history = pd.DataFrame(

            self.position_history

        )

        self.equity_curve = pd.DataFrame(

            self.equity_curve

        )

        if not self.equity_curve.empty:

            self.equity_curve.set_index(

                "Date",

                inplace=True,

            )

        self.logger.info(

            "Backtest data finalized."

        )

    # =========================================================
    # PERFORMANCE ANALYTICS
    # =========================================================

    def calculate_performance(self):

        self.logger.info(

            "Calculating portfolio performance..."

        )

        if self.equity_curve.empty:

            self.logger.warning(

                "Equity curve is empty."

            )

            return

        equity = self.equity_curve["Equity"].copy()

        returns = equity.pct_change().fillna(0.0)

        self.performance = self.metrics.calculate(

            equity_curve=equity,

            returns=returns,

            trades=self.trade_history,

        )

        self.logger.info(

            "Performance metrics calculated."

        )

    # =========================================================
    # BENCHMARK PERFORMANCE
    # =========================================================

    def calculate_benchmark(self):

        self.logger.info(

            "Calculating benchmark comparison..."

        )

        try:

            benchmark = self.downloader.download(

                symbols=[self.config.benchmark],

                start=self.config.start_date,

                end=self.config.end_date,

                interval=self.config.interval,

                use_cache=getattr(

                    self.config,

                    "cache_enabled",

                    False,

                ),

            )

            benchmark = benchmark[

                self.config.benchmark

            ]

            benchmark_returns = (

                benchmark["Close"]

                .pct_change()

                .fillna(0.0)

            )

            self.performance[

                "benchmark_return"

            ] = (

                benchmark_returns.add(1)

                .prod()

                - 1

            )

        except Exception as exc:

            self.logger.warning(

                f"Benchmark unavailable: {exc}"

            )

    # =========================================================
    # PERFORMANCE PLOTS
    # =========================================================

    def generate_plots(self):

        self.logger.info(

            "Generating plots..."

        )

        try:

            self.plots.generate_all(

                equity_curve=self.equity_curve,

                trades=self.trade_history,

                portfolio=self.portfolio_history,

            )

        except AttributeError:

            try:

                self.plots.plot_all(

                    self.equity_curve,

                    self.trade_history,

                )

            except Exception as exc:

                self.logger.warning(

                    f"Plot generation failed: {exc}"

                )

    # =========================================================
    # TEAR SHEET
    # =========================================================

    def generate_tearsheet(self):

        self.logger.info(

            "Generating tear sheet..."

        )

        try:

            self.report = self.tearsheet.generate(

                metrics=self.performance,

                equity_curve=self.equity_curve,

                trades=self.trade_history,

                portfolio=self.portfolio_history,

            )

        except AttributeError:

            try:

                self.report = self.tearsheet.create(

                    self.performance,

                    self.trade_history,

                    self.equity_curve,

                )

            except Exception as exc:

                self.logger.warning(

                    f"Tear sheet failed: {exc}"

                )

    # =========================================================
    # TRADE STATISTICS
    # =========================================================

    def calculate_trade_statistics(self):

        if self.trade_history.empty:

            return

        stats = {}

        stats["total_trades"] = len(

            self.trade_history

        )

        if "PnL" in self.trade_history.columns:

            pnl = self.trade_history["PnL"]

            stats["gross_profit"] = pnl[

                pnl > 0

            ].sum()

            stats["gross_loss"] = pnl[

                pnl < 0

            ].sum()

            stats["winning_trades"] = int(

                (pnl > 0).sum()

            )

            stats["losing_trades"] = int(

                (pnl < 0).sum()

            )

            stats["average_trade"] = float(

                pnl.mean()

            )

            stats["largest_win"] = float(

                pnl.max()

            )

            stats["largest_loss"] = float(

                pnl.min()

            )

        self.performance.update(

            stats

        )

    # =========================================================
    # EQUITY STATISTICS
    # =========================================================

    def calculate_equity_statistics(self):

        if self.equity_curve.empty:

            return

        equity = self.equity_curve["Equity"]

        self.performance["initial_equity"] = float(

            equity.iloc[0]

        )

        self.performance["final_equity"] = float(

            equity.iloc[-1]

        )

        self.performance["net_profit"] = float(

            equity.iloc[-1]

            - equity.iloc[0]

        )

        self.performance["return_percent"] = (

            self.performance["net_profit"]

            / equity.iloc[0]

        ) * 100.0

        running_max = equity.cummax()

        drawdown = (

            equity

            - running_max

        ) / running_max

        self.performance["max_drawdown"] = float(

            drawdown.min()

        )

    # =========================================================
    # ANALYTICS PIPELINE
    # =========================================================

    def analyze_results(self):

        self.calculate_performance()

        self.calculate_trade_statistics()

        self.calculate_equity_statistics()

        self.calculate_benchmark()

        self.generate_plots()

        self.generate_tearsheet()

        self.logger.info(

            "Analytics pipeline completed."

        )

    # =========================================================
    # EXPORT TRADES
    # =========================================================

    def export_trades(self):

        if self.trade_history.empty:

            self.logger.warning(

                "No trades to export."

            )

            return

        output = (

            Path(self.config.output_directory)

            / "trades.csv"

        )

        self.trade_history.to_csv(

            output,

            index=False,

        )

        self.logger.info(

            f"Trades exported -> {output}"

        )

    # =========================================================
    # EXPORT EQUITY CURVE
    # =========================================================

    def export_equity_curve(self):

        if self.equity_curve.empty:

            return

        output = (

            Path(self.config.output_directory)

            / "equity_curve.csv"

        )

        self.equity_curve.to_csv(

            output,

        )

        self.logger.info(

            f"Equity curve exported -> {output}"

        )

    # =========================================================
    # EXPORT PORTFOLIO HISTORY
    # =========================================================

    def export_portfolio_history(self):

        if self.portfolio_history.empty:

            return

        output = (

            Path(self.config.output_directory)

            / "portfolio_history.csv"

        )

        self.portfolio_history.to_csv(

            output,

            index=False,

        )

        self.logger.info(

            f"Portfolio history exported -> {output}"

        )

    # =========================================================
    # EXPORT POSITION HISTORY
    # =========================================================

    def export_position_history(self):

        if self.position_history.empty:

            return

        output = (

            Path(self.config.output_directory)

            / "position_history.csv"

        )

        self.position_history.to_csv(

            output,

            index=False,

        )

        self.logger.info(

            f"Position history exported -> {output}"

        )

    # =========================================================
    # EXPORT METRICS
    # =========================================================

    def export_metrics(self):

        if not self.performance:

            return

        output = (

            Path(self.config.output_directory)

            / "performance_metrics.csv"

        )

        metrics = pd.DataFrame(

            [

                {

                    "Metric": k,

                    "Value": v,

                }

                for k, v in self.performance.items()

            ]

        )

        metrics.to_csv(

            output,

            index=False,

        )

        self.logger.info(

            f"Metrics exported -> {output}"

        )

    # =========================================================
    # SAVE TEAR SHEET
    # =========================================================

    def save_tearsheet(self):

        if self.report is None:

            return

        try:

            output = (

                Path(self.config.output_directory)

                / "tearsheet.html"

            )

            self.tearsheet.save(

                self.report,

                output,

            )

            self.logger.info(

                f"Tear sheet saved -> {output}"

            )

        except AttributeError:

            pass

    # =========================================================
    # EXPORT ALL RESULTS
    # =========================================================

    def export_results(self):

        self.logger.info(

            "Exporting results..."

        )

        self.export_trades()

        self.export_equity_curve()

        self.export_portfolio_history()

        self.export_position_history()

        self.export_metrics()

        self.save_tearsheet()

        self.logger.info(

            "Export completed."

        )

    # =========================================================
    # SUMMARY REPORT
    # =========================================================

    def print_summary(self):

        self.logger.info(

            "=" * 70

        )

        self.logger.info(

            "BACKTEST SUMMARY"

        )

        self.logger.info(

            "=" * 70

        )

        for key, value in self.performance.items():

            self.logger.info(

                f"{key:30} : {value}"

            )

        self.logger.info(

            "=" * 70

        )

        self.logger.info(

            f"Trades Executed : {len(self.trade_history)}"

        )

        self.logger.info(

            f"Errors Logged   : {len(self.state.errors)}"

        )

        self.logger.info(

            "=" * 70

        )

    # =========================================================
    # CLEANUP
    # =========================================================

    def cleanup(self):

        self.logger.info(

            "Cleaning up resources..."

        )

        self.market_data.clear()

        self.cleaned_data.clear()

        self.indicator_data.clear()

        self.factor_data.clear()

        self.signal_data.clear()

        self.trade_candidates = {}

        self.execution_queue = []

        self.logger.info(

            "Cleanup complete."

        )

    # =========================================================
    # RUN COMPLETE BACKTEST
    # =========================================================

    def run(self):

        self.start()

        try:

            # -----------------------------------------
            # PHASE 1
            # -----------------------------------------

            self.logger.info(

                "Phase 1 : Data Preparation"

            )

            self.prepare_data()

            # -----------------------------------------
            # PHASE 2
            # -----------------------------------------

            self.logger.info(

                "Phase 2 : Feature Engineering"

            )

            self.prepare_features()

            # -----------------------------------------
            # PHASE 3
            # -----------------------------------------

            self.logger.info(

                "Phase 3 : Strategy"

            )

            self.prepare_strategy()

            # -----------------------------------------
            # PHASE 4
            # -----------------------------------------

            self.logger.info(

                "Phase 4 : Event Driven Backtest"

            )

            self.run_backtest()

            self.finalize_backtest_data()

            # -----------------------------------------
            # PHASE 5
            # -----------------------------------------

            self.logger.info(

                "Phase 5 : Analytics"

            )

            self.analyze_results()

            # -----------------------------------------
            # PHASE 6
            # -----------------------------------------

            self.logger.info(

                "Phase 6 : Export"

            )

            self.export_results()

            self.print_summary()

        except KeyboardInterrupt:

            self.logger.warning(

                "Backtest interrupted by user."

            )

            raise

        except Exception as exc:

            self.logger.exception(

                f"Backtest failed : {exc}"

            )

            raise

        finally:

            self.cleanup()

            self.stop()

    # =========================================================
    # RESET ENGINE
    # =========================================================

    def reset(self):

        self.logger.info(

            "Resetting engine..."

        )

        self.state = EngineState()

        self._initialize_runtime()

    # =========================================================
    # ENGINE STATUS
    # =========================================================

    @property
    def status(self):

        return {

            "running": self.state.running,

            "completed": self.state.completed,

            "processed_symbols": self.state.processed_symbols,

            "total_symbols": self.state.total_symbols,

            "errors": len(

                self.state.errors

            ),

        }

    # =========================================================
    # GET EQUITY CURVE
    # =========================================================

    def get_equity_curve(self):

        return self.equity_curve.copy()

    # =========================================================
    # GET TRADES
    # =========================================================

    def get_trades(self):

        return self.trade_history.copy()

    # =========================================================
    # GET PORTFOLIO
    # =========================================================

    def get_portfolio_history(self):

        return self.portfolio_history.copy()

    # =========================================================
    # GET METRICS
    # =========================================================

    def get_metrics(self):

        return dict(

            self.performance

        )

    # =========================================================
    # GET ERRORS
    # =========================================================

    def get_errors(self):

        return list(

            self.state.errors

        )


# =============================================================
# STANDALONE EXECUTION
# =============================================================

if __name__ == "__main__":

    engine = BacktestEngine()

    engine.run()
