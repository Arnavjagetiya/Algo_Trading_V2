"""
config.py
=============================================================
Global configuration for the Algo Trading Framework.

Author : Arnav
Python : 3.12+
=============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ============================================================
# PROJECT
# ============================================================

PROJECT_NAME: str = "Algo_Trading_V2"
VERSION: str = "1.0.0"

# ============================================================
# ROOT PATHS
# ============================================================

ROOT_DIR: Path = Path(__file__).resolve().parent

DATA_DIR: Path = ROOT_DIR / "data"
CACHE_DIR: Path = ROOT_DIR / "cache"
LOG_DIR: Path = ROOT_DIR / "logs"
OUTPUT_DIR: Path = ROOT_DIR / "output"
REPORT_DIR: Path = OUTPUT_DIR / "reports"
PLOT_DIR: Path = OUTPUT_DIR / "plots"

for directory in (
    DATA_DIR,
    CACHE_DIR,
    LOG_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    PLOT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# MARKET
# ============================================================

MARKET: Literal["NSE"] = "NSE"

TIMEZONE: str = "Asia/Kolkata"

BENCHMARK: str = "^NSEI"

RISK_FREE_RATE: float = 0.065

# ============================================================
# DATA
# ============================================================

DEFAULT_START_DATE: str = "2026-01-01"

DEFAULT_END_DATE: str | None = None

DEFAULT_INTERVAL: str = "1d"

INTRADAY_INTERVAL: str = "5m"

INTRADAY_PERIOD: str = "60d"

AUTO_ADJUST: bool = False

DOWNLOAD_THREADS: int = 8

DOWNLOAD_RETRIES: int = 5

REQUEST_TIMEOUT: int = 30

SLEEP_BETWEEN_RETRIES: float = 2.0

SHOW_PROGRESS: bool = True

ROUND_DECIMALS: int = 6

# ============================================================
# CACHE
# ============================================================

ENABLE_CACHE: bool = True

CACHE_FORMAT: Literal["parquet", "csv"] = "parquet"

CACHE_EXPIRE_HOURS: int = 12

# ============================================================
# VALIDATION
# ============================================================

REMOVE_DUPLICATES: bool = True

DROP_EMPTY_ROWS: bool = True

FORWARD_FILL: bool = False

BACKWARD_FILL: bool = False

REMOVE_WEEKENDS: bool = True

CHECK_NEGATIVE_PRICES: bool = True

CHECK_ZERO_VOLUME: bool = False

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL: Literal[
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
] = "INFO"

LOG_FILE: Path = LOG_DIR / "framework.log"

# ============================================================
# INDICATORS
# ============================================================

EMA_FAST: int = 20

EMA_SLOW: int = 50

EMA_LONG: int = 200

RSI_LENGTH: int = 14

ATR_LENGTH: int = 14

BB_LENGTH: int = 20

BB_STD: float = 2.0

MACD_FAST: int = 12

MACD_SLOW: int = 26

MACD_SIGNAL: int = 9

VWAP_ENABLED: bool = True

# ============================================================
# GARCH
# ============================================================

GARCH_WINDOW: int = 252

GARCH_P: int = 1

GARCH_Q: int = 3

GARCH_MEAN: str = "Zero"

GARCH_VOL: str = "GARCH"

GARCH_DIST: str = "normal"

# ============================================================
# PORTFOLIO
# ============================================================

INITIAL_CAPITAL: float = 100000.0

MAX_OPEN_POSITIONS: int = 10

POSITION_SIZE: float = 0.10

MAX_PORTFOLIO_EXPOSURE: float = 1.0

ALLOW_SHORTS: bool = True

# ============================================================
# RISK MANAGEMENT
# ============================================================

RISK_PER_TRADE: float = 0.01

ATR_STOP_MULTIPLIER: float = 1.5

ATR_TARGET_MULTIPLIER: float = 3.0

TRAILING_STOP_MULTIPLIER: float = 2.0

MAX_DAILY_LOSS: float = 0.03

MAX_DRAWDOWN: float = 0.20

# ============================================================
# TRANSACTION COSTS
# ============================================================

BROKERAGE_PER_ORDER: float = 20.0

SLIPPAGE_PERCENT: float = 0.0005

STT_PERCENT: float = 0.00025

EXCHANGE_CHARGE_PERCENT: float = 0.0000345

GST_PERCENT: float = 0.18

STAMP_DUTY_PERCENT: float = 0.00003

SEBI_CHARGE_PERCENT: float = 0.000001

# ============================================================
# BACKTEST
# ============================================================

REBALANCE_FREQUENCY: Literal[
    "daily",
    "weekly",
    "monthly",
] = "monthly"

LOOKBACK_MONTHS: int = 12

TOP_N_STOCKS: int = 15

MIN_HISTORY_DAYS: int = 252

# ============================================================
# OPTIMIZATION
# ============================================================

OPTIMIZATION_ENABLED: bool = False

RANDOM_SEED: int = 42

# ============================================================
# REPORTS
# ============================================================

SAVE_REPORTS: bool = True

SAVE_PLOTS: bool = True

EXPORT_TRADES: bool = True

EXPORT_EQUITY_CURVE: bool = True

# ============================================================
# DATACLASS CONFIG
# ============================================================


@dataclass(slots=True)
class FrameworkConfig:
    project_name: str = PROJECT_NAME
    version: str = VERSION

    benchmark: str = BENCHMARK

    market: str = MARKET
    timezone: str = TIMEZONE

    start_date: str = DEFAULT_START_DATE
    end_date: str | None = DEFAULT_END_DATE

    interval: str = DEFAULT_INTERVAL
    intraday_interval: str = INTRADAY_INTERVAL
    intraday_period: str = INTRADAY_PERIOD

    auto_adjust: bool = AUTO_ADJUST

    cache_enabled: bool = ENABLE_CACHE

    initial_capital: float = INITIAL_CAPITAL

    risk_per_trade: float = RISK_PER_TRADE

    random_seed: int = RANDOM_SEED

    data_directory: Path = field(default_factory=lambda: DATA_DIR)
    cache_directory: Path = field(default_factory=lambda: CACHE_DIR)
    log_directory: Path = field(default_factory=lambda: LOG_DIR)
    output_directory: Path = field(default_factory=lambda: OUTPUT_DIR)


CONFIG = FrameworkConfig()