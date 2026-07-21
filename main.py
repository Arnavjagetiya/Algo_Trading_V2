"""
main.py
============================================================
Professional Algo Trading Framework Entry Point
============================================================
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

from config import CONFIG


# ============================================================
# BANNER
# ============================================================

def get_config_value(name, fallback=None):

    if hasattr(CONFIG, name):

        return getattr(CONFIG, name)

    return getattr(CONFIG, name.upper(), fallback)


# ============================================================
# BANNER
# ============================================================

def print_banner():

    project_name = get_config_value("project_name", "Algo_Trading_V2")
    version = get_config_value("version", "unknown")
    market = get_config_value("market", "NSE")
    benchmark = get_config_value("benchmark", "^NSEI")

    print("=" * 80)
    print(f"{project_name}  v{version}")
    print("=" * 80)
    print(f"Started   : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Market    : {market}")
    print(f"Benchmark : {benchmark}")
    print("=" * 80)


# ============================================================
# DEPENDENCIES
# ============================================================

def check_dependencies():

    packages = [
        "numpy",
        "pandas",
        "yfinance",
        "matplotlib",
        "scipy",
        "sklearn",
        "statsmodels",
        "cvxpy",
        "arch",
        "pandas_ta",
    ]

    missing = []

    for package in packages:

        try:
            importlib.import_module(package)

        except ImportError:
            missing.append(package)

    if missing:

        print("\nMissing packages:\n")

        for package in missing:
            print(f"  - {package}")

        print("\nInstall them before running.\n")

        sys.exit(1)


# ============================================================
# DIRECTORIES
# ============================================================

def ensure_directories():

    directories = [

        get_config_value("data_directory"),

        get_config_value("cache_directory"),

        get_config_value("log_directory"),

        get_config_value("output_directory"),

    ]

    for directory in directories:

        Path(directory).mkdir(

            parents=True,

            exist_ok=True,

        )


# ============================================================
# LOGGER
# ============================================================

def configure_logging(debug=False):

    level = logging.DEBUG if debug else logging.INFO

    root_logger = logging.getLogger()

    if root_logger.handlers:

        root_logger.setLevel(level)
        return

    logging.basicConfig(

        level=level,

        format="%(asctime)s | %(levelname)s | %(message)s",

    )


# ============================================================
# ENGINE
# ============================================================

def create_engine():

    from backtest_engine import BacktestEngine

    return BacktestEngine(CONFIG)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--debug",

        action="store_true",

    )

    args = parser.parse_args()

    configure_logging(

        debug=args.debug,

    )

    print_banner()

    check_dependencies()

    ensure_directories()

    engine = create_engine()

    engine.run()

    print("\n")

    print("=" * 80)

    print("BACKTEST COMPLETED")

    print("=" * 80)

    try:

        metrics = engine.get_metrics()

        for key, value in metrics.items():

            print(f"{key:30} : {value}")

    except Exception:

        pass


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\nInterrupted by user.")

    except Exception:

        print("\nFramework crashed.\n")

        traceback.print_exc()

        sys.exit(1)