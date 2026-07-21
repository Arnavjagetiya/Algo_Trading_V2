"""
strategy/execution.py
=========================================================
Order Execution Engine

Author : Arnav
Python : 3.12+
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# ENUMS
# ============================================================


class OrderSide(str, Enum):

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):

    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


# ============================================================
# ORDER
# ============================================================


@dataclass(slots=True)
class Order:

    ticker: str

    quantity: float

    side: OrderSide

    order_type: OrderType = OrderType.MARKET

    limit_price: Optional[float] = None

    stop_price: Optional[float] = None

    timestamp: Optional[pd.Timestamp] = None

    status: OrderStatus = OrderStatus.PENDING

    fill_price: Optional[float] = None

    commission: float = 0.0

    slippage: float = 0.0

    order_id: int = 0


# ============================================================
# TRADE
# ============================================================


@dataclass(slots=True)
class Trade:

    ticker: str

    side: OrderSide

    quantity: float

    entry_price: float

    exit_price: float

    pnl: float

    commission: float

    entry_time: pd.Timestamp

    exit_time: pd.Timestamp


# ============================================================
# EXECUTION ENGINE
# ============================================================


class ExecutionEngine:

    """
    Simulated execution engine.

    Supports

    ✓ Market Orders
    ✓ Limit Orders
    ✓ Stop Orders
    ✓ Slippage
    ✓ Commission
    """

    def __init__(

        self,

        commission_rate: float = 0.0005,

        slippage_rate: float = 0.0005,

    ):

        self.commission_rate = commission_rate

        self.slippage_rate = slippage_rate

        self.orders: list[Order] = []

        self.trades: list[Trade] = []

        self.order_counter = 1

    # ========================================================
    # ORDER VALIDATION
    # ========================================================

    @staticmethod
    def validate(order: Order):

        if order.quantity <= 0:

            raise ValueError(

                "Quantity must be positive."

            )

        if order.side not in (

            OrderSide.BUY,

            OrderSide.SELL,

        ):

            raise ValueError(

                "Invalid order side."

            )

        if order.order_type == OrderType.LIMIT:

            if order.limit_price is None:

                raise ValueError(

                    "Limit price required."

                )

        if order.order_type == OrderType.STOP:

            if order.stop_price is None:

                raise ValueError(

                    "Stop price required."

                )

    # ========================================================
    # COMMISSION
    # ========================================================

    def commission(

        self,

        quantity,

        price,

    ):

        return (

            quantity

            * price

            * self.commission_rate

        )

    # ========================================================
    # SLIPPAGE
    # ========================================================

    def slippage(

        self,

        price,

        side,

    ):

        if side == OrderSide.BUY:

            return (

                price

                * (

                    1

                    + self.slippage_rate

                )

            )

        return (

            price

            * (

                1

                - self.slippage_rate

            )

        )

    # ========================================================
    # PLACE ORDER
    # ========================================================

    def place_order(

        self,

        order: Order,

    ):

        self.validate(order)

        order.order_id = self.order_counter

        self.order_counter += 1

        self.orders.append(order)

        return order.order_id

    # ========================================================
    # CANCEL
    # ========================================================

    def cancel_order(

        self,

        order_id,

    ):

        for order in self.orders:

            if order.order_id == order_id:

                if order.status == OrderStatus.PENDING:

                    order.status = OrderStatus.CANCELLED

                    return True

        return False

    # ========================================================
    # EXECUTION
    # ========================================================

    def execute(

        self,

        order: Order | None = None,

        candle: pd.Series | None = None,

        symbol: str | None = None,

        signal: int | None = None,

        quantity: float | None = None,

        price: float | None = None,

        timestamp: pd.Timestamp | None = None,

    ):

        if order is None:

            return {

                "symbol": symbol or "",

                "quantity": quantity or 0,

                "signal": signal or 0,

                "price": price,

                "timestamp": timestamp,

            }

        if candle is None and price is not None:

            candle = pd.Series({"Open": price, "Low": price, "High": price})

        if order.status != OrderStatus.PENDING:

            return order

        fill = None

        # --------------------------------------------
        # MARKET
        # --------------------------------------------

        if order.order_type == OrderType.MARKET:

            fill = candle["Open"]

        # --------------------------------------------
        # LIMIT
        # --------------------------------------------

        elif order.order_type == OrderType.LIMIT:

            if (

                candle["Low"]

                <= order.limit_price

                <= candle["High"]

            ):

                fill = order.limit_price

        # --------------------------------------------
        # STOP
        # --------------------------------------------

        elif order.order_type == OrderType.STOP:

            if order.side == OrderSide.BUY:

                if candle["High"] >= order.stop_price:

                    fill = order.stop_price

            else:

                if candle["Low"] <= order.stop_price:

                    fill = order.stop_price

        if fill is None:

            return order

        fill = self.slippage(

            fill,

            order.side,

        )

        comm = self.commission(

            order.quantity,

            fill,

        )

        order.fill_price = fill

        order.commission = comm

        order.slippage = abs(

            fill

            - candle["Open"]

        )

        order.status = OrderStatus.FILLED

        return order
    
    # ========================================================
    # ORDER BOOK
    # ========================================================

    def pending_orders(
        self,
    ) -> list[Order]:

        return [
            order
            for order in self.orders
            if order.status == OrderStatus.PENDING
        ]

    # --------------------------------------------------------

    def filled_orders(
        self,
    ) -> list[Order]:

        return [
            order
            for order in self.orders
            if order.status == OrderStatus.FILLED
        ]

    # --------------------------------------------------------

    def cancelled_orders(
        self,
    ) -> list[Order]:

        return [
            order
            for order in self.orders
            if order.status == OrderStatus.CANCELLED
        ]

    # ========================================================
    # POSITION MANAGEMENT
    # ========================================================

    def open_position(
        self,
        ticker: str,
    ) -> bool:

        qty = 0.0

        for order in self.filled_orders():

            if order.ticker != ticker:
                continue

            if order.side == OrderSide.BUY:
                qty += order.quantity
            else:
                qty -= order.quantity

        return qty != 0

    # --------------------------------------------------------

    def position_size(
        self,
        ticker: str,
    ) -> float:

        qty = 0.0

        for order in self.filled_orders():

            if order.ticker != ticker:
                continue

            if order.side == OrderSide.BUY:
                qty += order.quantity
            else:
                qty -= order.quantity

        return qty

    # --------------------------------------------------------

    def average_entry_price(
        self,
        ticker: str,
    ) -> float:

        buys = []

        qty = []

        for order in self.filled_orders():

            if order.ticker != ticker:
                continue

            if order.side == OrderSide.BUY:

                buys.append(order.fill_price)
                qty.append(order.quantity)

        if not buys:
            return 0.0

        return np.average(
            buys,
            weights=qty,
        )

    # ========================================================
    # CLOSE POSITION
    # ========================================================

    def close_position(
        self,
        ticker: str,
        exit_price: float,
        timestamp: pd.Timestamp,
    ):

        position = self.position_size(ticker)

        if position == 0:
            return None

        entry = self.average_entry_price(
            ticker
        )

        commission = self.commission(
            abs(position),
            exit_price,
        )

        pnl = (
            exit_price - entry
        ) * position - commission

        trade = Trade(

            ticker=ticker,

            side=(
                OrderSide.BUY
                if position > 0
                else OrderSide.SELL
            ),

            quantity=abs(position),

            entry_price=entry,

            exit_price=exit_price,

            pnl=pnl,

            commission=commission,

            entry_time=pd.NaT,

            exit_time=timestamp,

        )

        self.trades.append(trade)

        side = (
            OrderSide.SELL
            if position > 0
            else OrderSide.BUY
        )

        close_order = Order(

            ticker=ticker,

            quantity=abs(position),

            side=side,

            order_type=OrderType.MARKET,

            timestamp=timestamp,

            status=OrderStatus.FILLED,

            fill_price=exit_price,

            commission=commission,

            order_id=self.order_counter,

        )

        self.order_counter += 1

        self.orders.append(close_order)

        return trade

    # ========================================================
    # EXECUTE BAR
    # ========================================================

    def process_bar(
        self,
        candle: pd.Series,
    ):

        for order in self.pending_orders():

            self.execute(
                order,
                candle,
            )

    # ========================================================
    # CASH
    # ========================================================

    @staticmethod
    def cash_after_buy(
        cash: float,
        quantity: float,
        price: float,
        commission: float,
    ) -> float:

        return (
            cash
            - quantity * price
            - commission
        )

    # --------------------------------------------------------

    @staticmethod
    def cash_after_sell(
        cash: float,
        quantity: float,
        price: float,
        commission: float,
    ) -> float:

        return (
            cash
            + quantity * price
            - commission
        )

    # ========================================================
    # TRADEBOOK
    # ========================================================

    def tradebook(
        self,
    ) -> pd.DataFrame:

        if not self.trades:

            return pd.DataFrame()

        return pd.DataFrame(

            [

                {

                    "Ticker": t.ticker,

                    "Side": t.side.value,

                    "Quantity": t.quantity,

                    "Entry": t.entry_price,

                    "Exit": t.exit_price,

                    "PnL": t.pnl,

                    "Commission": t.commission,

                    "Entry Time": t.entry_time,

                    "Exit Time": t.exit_time,

                }

                for t in self.trades

            ]

        )

    # ========================================================
    # ORDERBOOK
    # ========================================================

    def orderbook(
        self,
    ) -> pd.DataFrame:

        if not self.orders:

            return pd.DataFrame()

        return pd.DataFrame(

            [

                {

                    "Order ID": o.order_id,

                    "Ticker": o.ticker,

                    "Side": o.side.value,

                    "Type": o.order_type.value,

                    "Qty": o.quantity,

                    "Status": o.status.value,

                    "Fill Price": o.fill_price,

                    "Commission": o.commission,

                    "Slippage": o.slippage,

                }

                for o in self.orders

            ]

        )
        
    # ========================================================
    # PORTFOLIO
    # ========================================================

    def total_commission(
        self,
    ) -> float:

        return float(
            sum(
                order.commission
                for order in self.orders
                if order.status == OrderStatus.FILLED
            )
        )

    # --------------------------------------------------------

    def total_slippage(
        self,
    ) -> float:

        return float(
            sum(
                order.slippage
                for order in self.orders
                if order.status == OrderStatus.FILLED
            )
        )

    # --------------------------------------------------------

    def total_pnl(
        self,
    ) -> float:

        return float(
            sum(
                trade.pnl
                for trade in self.trades
            )
        )

    # --------------------------------------------------------

    def win_rate(
        self,
    ) -> float:

        if not self.trades:
            return 0.0

        wins = sum(
            trade.pnl > 0
            for trade in self.trades
        )

        return wins / len(self.trades)

    # --------------------------------------------------------

    def average_win(
        self,
    ) -> float:

        wins = [
            trade.pnl
            for trade in self.trades
            if trade.pnl > 0
        ]

        if not wins:
            return 0.0

        return float(np.mean(wins))

    # --------------------------------------------------------

    def average_loss(
        self,
    ) -> float:

        losses = [
            trade.pnl
            for trade in self.trades
            if trade.pnl < 0
        ]

        if not losses:
            return 0.0

        return float(np.mean(losses))

    # --------------------------------------------------------

    def profit_factor(
        self,
    ) -> float:

        gross_profit = sum(
            trade.pnl
            for trade in self.trades
            if trade.pnl > 0
        )

        gross_loss = abs(
            sum(
                trade.pnl
                for trade in self.trades
                if trade.pnl < 0
            )
        )

        if gross_loss == 0:
            return np.inf

        return gross_profit / gross_loss

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ):

        self.orders.clear()

        self.trades.clear()

        self.order_counter = 1

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            {
                "Orders": [len(self.orders)],
                "Filled": [
                    len(self.filled_orders())
                ],
                "Cancelled": [
                    len(self.cancelled_orders())
                ],
                "Trades": [len(self.trades)],
                "Net PnL": [
                    self.total_pnl()
                ],
                "Commission": [
                    self.total_commission()
                ],
                "Slippage": [
                    self.total_slippage()
                ],
                "Win Rate": [
                    self.win_rate()
                ],
                "Average Win": [
                    self.average_win()
                ],
                "Average Loss": [
                    self.average_loss()
                ],
                "Profit Factor": [
                    self.profit_factor()
                ],
            }
        )

    # ========================================================
    # EXPORTS
    # ========================================================

    def export_orders(
        self,
        filepath: str,
    ):

        self.orderbook().to_csv(
            filepath,
            index=False,
        )

    # --------------------------------------------------------

    def export_trades(
        self,
        filepath: str,
    ):

        self.tradebook().to_csv(
            filepath,
            index=False,
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    def info(
        self,
    ) -> dict:

        return {

            "Orders": len(self.orders),

            "Pending": len(
                self.pending_orders()
            ),

            "Filled": len(
                self.filled_orders()
            ),

            "Cancelled": len(
                self.cancelled_orders()
            ),

            "Trades": len(self.trades),

            "Commission Rate": self.commission_rate,

            "Slippage Rate": self.slippage_rate,

        }

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ):

        return (

            f"{self.__class__.__name__}("
            f"orders={len(self.orders)}, "
            f"trades={len(self.trades)}, "
            f"commission={self.commission_rate:.4f}, "
            f"slippage={self.slippage_rate:.4f})"

        )
