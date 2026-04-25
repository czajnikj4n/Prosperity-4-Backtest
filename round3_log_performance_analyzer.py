#!/usr/bin/env python3
"""
Round 3 Prosperity log analyzer for VELVETFRUIT / VEV vouchers / HYDROGEL.

Purpose
-------
Reconstruct actual SUBMISSION fills from a Prosperity backtest log and produce:

1. Per-product realized / unrealized / total PnL.
2. Position, turnover, drawdown, inventory efficiency and trade edge metrics.
3. Portfolio-level PnL across all Round 3 products.
4. Option/voucher diagnostics for VEV_xxxx products:
   - Black-Scholes theoretical value from fitted vol smile
   - delta / vega
   - theoretical diff = market_mid - theoretical
   - EMA-normalized theoretical diff
   - adjusted fair value = theoretical + EMA(theoretical diff)
   - trade edge versus adjusted fair
5. Threshold sweeps to help tune entry parameters.

Expected log structure
----------------------
The analyzer expects a Prosperity JSON log containing:
    - activitiesLog: semicolon-separated market table
    - tradeHistory: actual trades
    - logs: optional warnings/log output

Usage
-----
Set LOG_PATH at the bottom and run this file.

    python round3_log_performance_analyzer.py

Or from another script:

    analyzer = Round3LogAnalyzer("/path/to/log.log")
    analyzer.load_all()
    analyzer.run_group_report()

Notes
-----
This does not rerun your strategy. It analyzes the fills/logs you already have.
The option diagnostics reconstruct the same kind of theoretical signal used by
your new Trader class so you can see whether the bot bought cheap / sold rich.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SUBMISSION = "SUBMISSION"

UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDROGEL = "HYDROGEL_PACK"

VOUCHER_STRIKES: Dict[str, int] = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}

ROUND3_PRODUCTS: List[str] = [
    HYDROGEL,
    UNDERLYING,
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]

ACTIVE_VOUCHERS: List[str] = [
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
]

# Same smile as in your strategy.
IV_SMILE_COEFFS = [0.27362531, 0.01007566, 0.14876677]

DAYS_PER_YEAR = 250
ROUND_DAY = 3

THEO_NORM_WINDOW = 500
IV_SCALPING_WINDOW = 500


@dataclass
class PositionSummary:
    product: str
    product_group: str
    n_market_rows: int
    n_submission_trades: int

    final_position: float
    final_cash: float
    final_mid: float
    final_total_pnl: float
    final_realized_pnl: float
    final_unrealized_pnl: float

    max_abs_position: float
    avg_abs_position: float
    turnover: float
    pnl_per_turnover: float

    max_drawdown: float
    pnl_volatility: float
    inventory_efficiency: float

    first_reach_25pct_idx: Optional[int]
    first_reach_50pct_idx: Optional[int]
    first_reach_75pct_idx: Optional[int]
    first_reach_90pct_idx: Optional[int]

    avg_buy_edge_vs_mid: Optional[float]
    avg_sell_edge_vs_mid: Optional[float]
    avg_trade_edge_vs_mid: Optional[float]

    avg_buy_edge_vs_fair: Optional[float]
    avg_sell_edge_vs_fair: Optional[float]
    avg_trade_edge_vs_fair: Optional[float]

    peak_unrealized_pnl: float
    final_unrealized_giveback: float

    largest_single_buy_qty: float
    largest_single_sell_qty: float
    largest_position_drop: float
    largest_position_rise: float


@dataclass
class OptionSignalSummary:
    product: str
    strike: int
    n_rows: int

    final_mid: float
    final_theo: float
    final_adjusted_fair: float
    final_theo_diff: float
    final_mean_theo_diff: float
    final_residual: float

    avg_mid: float
    avg_theo: float
    avg_adjusted_fair: float
    avg_theo_diff: float
    avg_abs_residual: float
    avg_absdev_ema: float

    avg_delta: float
    avg_vega: float
    avg_spread: float

    buy_signal_rows_thr_1: int
    sell_signal_rows_thr_1: int

    premium_predicts_h10_ic: Optional[float]
    premium_predicts_h100_ic: Optional[float]


class Round3LogAnalyzer:
    """
    Reconstruct actual position / PnL from Prosperity logs and add Round 3
    voucher-specific diagnostics.
    """

    def __init__(
        self,
        log_file_path: str,
        products: Optional[List[str]] = None,
        submission_name: str = SUBMISSION,
    ):
        self.log_file_path = log_file_path
        self.products = products or ROUND3_PRODUCTS
        self.submission_name = submission_name

        self.data: Optional[dict] = None
        self.activities_df: Optional[pd.DataFrame] = None
        self.trades_df: Optional[pd.DataFrame] = None
        self.logs_df: Optional[pd.DataFrame] = None

        self._option_signal_cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        with open(self.log_file_path, "r") as f:
            content = f.read().strip()

        try:
            self.data = json.loads(content)
        except json.JSONDecodeError:
            # Some Prosperity exports place JSON on the first line.
            self.data = json.loads(content.splitlines()[0])

    def parse_activities(self) -> None:
        if self.data is None:
            raise ValueError("Call load() first.")

        raw_csv = self.data.get("activitiesLog", "")
        if not raw_csv:
            self.activities_df = pd.DataFrame()
            return

        self.activities_df = pd.read_csv(io.StringIO(raw_csv), sep=";")

    def parse_trades(self) -> None:
        if self.data is None:
            raise ValueError("Call load() first.")
        self.trades_df = pd.DataFrame(self.data.get("tradeHistory", []))

    def parse_logs(self) -> None:
        if self.data is None:
            raise ValueError("Call load() first.")
        self.logs_df = pd.DataFrame(self.data.get("logs", []))

    def load_all(self) -> None:
        self.load()
        self.parse_activities()
        self.parse_trades()
        self.parse_logs()
        self.preprocess()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self) -> None:
        self._preprocess_activities()
        self._preprocess_trades()
        self._preprocess_logs()
        self._option_signal_cache.clear()

    def _preprocess_activities(self) -> None:
        if self.activities_df is None or self.activities_df.empty:
            raise ValueError("activities_df is empty. Call parse_activities() first.")

        df = self.activities_df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        numeric_cols = [
            "day", "timestamp",
            "bid_price_1", "bid_volume_1",
            "bid_price_2", "bid_volume_2",
            "bid_price_3", "bid_volume_3",
            "ask_price_1", "ask_volume_1",
            "ask_price_2", "ask_volume_2",
            "ask_price_3", "ask_volume_3",
            "mid_price", "profit_and_loss",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "product" not in df.columns:
            raise ValueError("activities_df must contain a product column.")

        # Ensure missing book columns exist.
        for col in [
            "bid_price_1", "bid_volume_1", "ask_price_1", "ask_volume_1",
            "profit_and_loss",
        ]:
            if col not in df.columns:
                df[col] = np.nan

        df = df.sort_values(["product", "day", "timestamp"], kind="stable").reset_index(drop=True)

        bid = df["bid_price_1"]
        ask = df["ask_price_1"]

        df["mid_price_calc"] = np.where(
            bid.notna() & ask.notna(),
            (bid + ask) / 2.0,
            np.where(bid.notna(), bid, np.where(ask.notna(), ask, np.nan)),
        )

        df["mid_price_calc"] = (
            df.groupby("product")["mid_price_calc"]
            .ffill()
            .bfill()
        )

        df["spread"] = df["ask_price_1"] - df["bid_price_1"]
        df["spread"] = df["spread"].where(df["spread"].notna(), 0.0)

        denom = df["bid_volume_1"].fillna(0) + df["ask_volume_1"].fillna(0)
        df["microprice"] = np.where(
            df["bid_price_1"].notna() & df["ask_price_1"].notna() & (denom > 0),
            (
                df["ask_price_1"] * df["bid_volume_1"].fillna(0)
                + df["bid_price_1"] * df["ask_volume_1"].fillna(0)
            ) / denom.replace(0, np.nan),
            np.nan,
        )

        df["microprice_offset"] = df["microprice"] - df["mid_price_calc"]
        df["book_imbalance"] = np.where(
            denom > 0,
            (df["bid_volume_1"].fillna(0) - df["ask_volume_1"].fillna(0)) / denom,
            0.0,
        )

        df["mid_diff"] = df.groupby("product")["mid_price_calc"].diff()
        df["row_id"] = np.arange(len(df), dtype=np.int64)

        df["day"] = df["day"].fillna(0).astype("int64")
        df["timestamp"] = df["timestamp"].fillna(0).astype("int64")

        df["event_time"] = (
            df["day"] * 1_000_000 + df["timestamp"]
        ).astype("int64")

        self.activities_df = df

    def _preprocess_trades(self) -> None:
        if self.trades_df is None:
            self.trades_df = pd.DataFrame()
            return

        if self.trades_df.empty:
            return

        trades = self.trades_df.copy()
        trades.columns = [str(c).strip() for c in trades.columns]

        for col in ["timestamp", "price", "quantity", "day"]:
            if col in trades.columns:
                trades[col] = pd.to_numeric(trades[col], errors="coerce")

        for col in ["buyer", "seller", "symbol", "currency"]:
            if col not in trades.columns:
                trades[col] = ""

        trades["buyer"] = trades["buyer"].fillna("").astype(str)
        trades["seller"] = trades["seller"].fillna("").astype(str)
        trades["symbol"] = trades["symbol"].fillna("").astype(str)

        if "day" not in trades.columns or trades["day"].isna().all():
            trades["day"] = self._infer_trade_days(trades)
        else:
            trades["day"] = trades["day"].fillna(0).astype("int64")

        trades["timestamp"] = trades["timestamp"].fillna(0).astype("int64")
        trades["price"] = trades["price"].astype(float)
        trades["quantity"] = trades["quantity"].astype(float)

        trades["event_time"] = (
            trades["day"] * 1_000_000 + trades["timestamp"]
        ).astype("int64")

        trades["signed_qty"] = np.where(
            trades["buyer"] == self.submission_name,
            trades["quantity"],
            np.where(trades["seller"] == self.submission_name, -trades["quantity"], 0.0),
        )

        trades["cash_flow"] = -trades["signed_qty"] * trades["price"]
        trades["abs_qty"] = trades["signed_qty"].abs()

        self.trades_df = (
            trades.sort_values(["symbol", "event_time"], kind="stable")
            .reset_index(drop=True)
        )

    def _infer_trade_days(self, trades: pd.DataFrame) -> np.ndarray:
        market_clock = (
            self.activities_df[["day", "timestamp", "event_time"]]
            .drop_duplicates(["day", "timestamp"])
            .sort_values("event_time")
            .reset_index(drop=True)
        )

        tmp = trades[["timestamp"]].copy()
        tmp["timestamp"] = pd.to_numeric(tmp["timestamp"], errors="coerce").fillna(0).astype("int64")
        tmp = tmp.sort_values("timestamp").reset_index()

        market_clock["timestamp"] = pd.to_numeric(
            market_clock["timestamp"], errors="coerce"
        ).fillna(0).astype("int64")

        inferred = pd.merge_asof(
            tmp.sort_values("timestamp"),
            market_clock[["timestamp", "day"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        out = pd.Series(index=tmp["index"], data=inferred["day"].fillna(0).astype("int64").values)
        return out.sort_index().to_numpy()

    def _preprocess_logs(self) -> None:
        if self.logs_df is None:
            self.logs_df = pd.DataFrame()
            return

        if self.logs_df.empty:
            return

        logs = self.logs_df.copy()

        if "timestamp" in logs.columns:
            logs["timestamp"] = pd.to_numeric(logs["timestamp"], errors="coerce")

        for col in ["sandboxLog", "lambdaLog"]:
            if col not in logs.columns:
                logs[col] = ""
            logs[col] = logs[col].fillna("").astype(str)

        self.logs_df = logs

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def product_group(self, product: str) -> str:
        if product == UNDERLYING:
            return "UNDERLYING"
        if product == HYDROGEL:
            return "REGULAR"
        if product in VOUCHER_STRIKES:
            return "VOUCHER"
        return "OTHER"

    def available_products(self) -> List[str]:
        if self.activities_df is None or self.activities_df.empty:
            return []
        return sorted(self.activities_df["product"].dropna().unique().tolist())

    def get_market(self, product: str) -> pd.DataFrame:
        if self.activities_df is None or self.activities_df.empty:
            return pd.DataFrame()

        return (
            self.activities_df[self.activities_df["product"] == product]
            .sort_values(["day", "timestamp"], kind="stable")
            .reset_index(drop=True)
            .copy()
        )

    def get_submission_trades(self, product: str) -> pd.DataFrame:
        if self.trades_df is None or self.trades_df.empty:
            return pd.DataFrame()

        out = self.trades_df[
            (self.trades_df["symbol"] == product)
            & (self.trades_df["signed_qty"] != 0)
        ].copy()

        return out.sort_values("event_time", kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Option math / signal reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def normal_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def get_tte(self, timestamp: float) -> float:
        progress = (
            DAYS_PER_YEAR
            - 8
            + ROUND_DAY
            + int(timestamp) // 100 / 10_000
        ) / DAYS_PER_YEAR

        return max(1.0 - progress, 1e-6)

    def get_iv(self, S: float, K: float, TTE: float) -> float:
        S = max(float(S), 1e-9)
        K = max(float(K), 1e-9)
        TTE = max(float(TTE), 1e-6)

        m_t_k = np.log(K / S) / math.sqrt(TTE)
        iv = float(np.poly1d(IV_SMILE_COEFFS)(m_t_k))
        return min(max(iv, 0.03), 1.50)

    def get_option_values(self, S: float, K: float, TTE: float) -> Tuple[float, float, float, float]:
        S = max(float(S), 1e-9)
        K = max(float(K), 1e-9)
        TTE = max(float(TTE), 1e-6)

        sigma = self.get_iv(S, K, TTE)
        sqrt_t = math.sqrt(TTE)

        d1 = (math.log(S / K) + 0.5 * sigma * sigma * TTE) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        call_value = S * self.normal_cdf(d1) - K * self.normal_cdf(d2)
        delta = self.normal_cdf(d1)
        vega = S * self.normal_pdf(d1) * sqrt_t

        return call_value, delta, vega, sigma

    def get_underlying_panel(self) -> pd.DataFrame:
        u = self.get_market(UNDERLYING)
        if u.empty:
            return pd.DataFrame()

        out = u[[
            "event_time", "day", "timestamp", "mid_price_calc",
            "microprice_offset", "book_imbalance"
        ]].copy()

        out = out.rename(columns={
            "mid_price_calc": "underlying_mid",
            "microprice_offset": "underlying_microprice_offset",
            "book_imbalance": "underlying_book_imbalance",
        })

        # Match the fair-value skew used in the strategy.
        out["underlying_fair"] = (
            out["underlying_mid"]
            + 0.30 * out["underlying_microprice_offset"].fillna(0.0)
            + 0.25 * out["underlying_book_imbalance"].fillna(0.0)
        )

        return out.sort_values("event_time").reset_index(drop=True)

    def get_option_signal_market(self, product: str) -> pd.DataFrame:
        """
        Market table with option theoretical signals attached.
        Only meaningful for VEV_xxxx products.
        """
        if product in self._option_signal_cache:
            return self._option_signal_cache[product].copy()

        market = self.get_market(product)

        if market.empty or product not in VOUCHER_STRIKES:
            self._option_signal_cache[product] = market
            return market.copy()

        underlying = self.get_underlying_panel()
        if underlying.empty:
            self._option_signal_cache[product] = market
            return market.copy()

        merged = pd.merge_asof(
            market.sort_values("event_time"),
            underlying[[
                "event_time", "underlying_mid", "underlying_fair",
                "underlying_microprice_offset", "underlying_book_imbalance",
            ]].sort_values("event_time"),
            on="event_time",
            direction="backward",
        )

        K = VOUCHER_STRIKES[product]

        theo_values = []
        deltas = []
        vegas = []
        ivs = []
        ttes = []

        for row in merged.itertuples(index=False):
            S = float(row.underlying_fair) if not pd.isna(row.underlying_fair) else np.nan
            TTE = self.get_tte(row.timestamp)

            if pd.isna(S):
                theo_values.append(np.nan)
                deltas.append(np.nan)
                vegas.append(np.nan)
                ivs.append(np.nan)
                ttes.append(TTE)
                continue

            value, delta, vega, iv = self.get_option_values(S, K, TTE)
            theo_values.append(value)
            deltas.append(delta)
            vegas.append(vega)
            ivs.append(iv)
            ttes.append(TTE)

        merged["strike"] = K
        merged["tte"] = ttes
        merged["iv"] = ivs
        merged["theo"] = theo_values
        merged["delta"] = deltas
        merged["vega"] = vegas

        merged["intrinsic"] = np.maximum(merged["underlying_fair"] - K, 0.0)
        merged["premium_to_intrinsic"] = merged["mid_price_calc"] - merged["intrinsic"]

        merged["theo_diff"] = merged["mid_price_calc"] - merged["theo"]

        # Online EMA approximation matching the Trader class.
        merged["mean_theo_diff"] = (
            merged["theo_diff"]
            .ewm(span=THEO_NORM_WINDOW, adjust=False)
            .mean()
        )

        merged["theo_residual"] = merged["theo_diff"] - merged["mean_theo_diff"]

        merged["residual_absdev_ema"] = (
            merged["theo_residual"].abs()
            .ewm(span=IV_SCALPING_WINDOW, adjust=False)
            .mean()
        )

        merged["adjusted_fair"] = merged["theo"] + merged["mean_theo_diff"]

        merged["buy_edge_vs_fair_at_ask"] = merged["adjusted_fair"] - merged["ask_price_1"]
        merged["sell_edge_vs_fair_at_bid"] = merged["bid_price_1"] - merged["adjusted_fair"]

        merged["premium_intrinsic_mean"] = (
            merged["premium_to_intrinsic"]
            .ewm(span=THEO_NORM_WINDOW, adjust=False)
            .mean()
        )

        merged["premium_intrinsic_residual"] = (
            merged["premium_to_intrinsic"] - merged["premium_intrinsic_mean"]
        )

        self._option_signal_cache[product] = merged
        return merged.copy()

    # ------------------------------------------------------------------
    # Accounting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _realized_unrealized_from_trades(trades: pd.DataFrame, mark_price: float) -> Dict[str, float]:
        """
        Average-cost accounting from actual fills.
        """
        pos = 0.0
        avg_cost = 0.0
        realized = 0.0

        for row in trades.itertuples(index=False):
            qty = float(row.signed_qty)
            px = float(row.price)

            if qty > 0:
                if pos >= 0:
                    new_pos = pos + qty
                    avg_cost = (avg_cost * pos + px * qty) / new_pos if new_pos != 0 else 0.0
                    pos = new_pos
                else:
                    cover = min(qty, -pos)
                    realized += (avg_cost - px) * cover
                    pos += cover
                    rem = qty - cover
                    if pos == 0:
                        avg_cost = 0.0
                    if rem > 0:
                        pos = rem
                        avg_cost = px

            elif qty < 0:
                sell_qty = -qty
                if pos <= 0:
                    new_short = (-pos) + sell_qty
                    avg_cost = (avg_cost * (-pos) + px * sell_qty) / new_short if new_short != 0 else 0.0
                    pos -= sell_qty
                else:
                    close = min(sell_qty, pos)
                    realized += (px - avg_cost) * close
                    pos -= close
                    rem = sell_qty - close
                    if pos == 0:
                        avg_cost = 0.0
                    if rem > 0:
                        pos = -rem
                        avg_cost = px

        if pos > 0:
            unrealized = (mark_price - avg_cost) * pos
        elif pos < 0:
            unrealized = (avg_cost - mark_price) * (-pos)
        else:
            unrealized = 0.0

        return {
            "position": pos,
            "avg_cost": avg_cost,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "total_pnl": realized + unrealized,
        }

    @staticmethod
    def _first_reach_index(series: pd.Series, threshold: float) -> Optional[int]:
        hit = series[series >= threshold]
        if hit.empty:
            return None
        return int(hit.index[0])

    @staticmethod
    def _max_drawdown(pnl_series: pd.Series) -> float:
        running_peak = pnl_series.cummax()
        dd = pnl_series - running_peak
        return float(dd.min())

    @staticmethod
    def _trade_edge_vs_mid(signed_qty: float, trade_price: float, mid_price: float) -> float:
        if signed_qty > 0:
            return mid_price - trade_price
        if signed_qty < 0:
            return trade_price - mid_price
        return np.nan

    @staticmethod
    def _trade_edge_vs_fair(signed_qty: float, trade_price: float, fair: float) -> float:
        if pd.isna(fair):
            return np.nan
        if signed_qty > 0:
            return fair - trade_price
        if signed_qty < 0:
            return trade_price - fair
        return np.nan

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def build_position_timeseries(self, product: str) -> pd.DataFrame:
        market = self.get_option_signal_market(product) if product in VOUCHER_STRIKES else self.get_market(product)

        if market.empty:
            raise ValueError(f"No market data for {product}")

        trades = self.get_submission_trades(product)

        base_cols = [
            "day", "timestamp", "row_id", "event_time",
            "mid_price_calc", "bid_price_1", "ask_price_1",
            "spread", "microprice_offset", "book_imbalance",
            "profit_and_loss",
        ]

        option_cols = [
            "underlying_mid", "underlying_fair", "strike", "tte", "iv",
            "theo", "delta", "vega", "intrinsic",
            "premium_to_intrinsic", "theo_diff", "mean_theo_diff",
            "theo_residual", "residual_absdev_ema", "adjusted_fair",
            "buy_edge_vs_fair_at_ask", "sell_edge_vs_fair_at_bid",
            "premium_intrinsic_residual",
        ]

        keep_cols = [c for c in base_cols + option_cols if c in market.columns]
        out = market[keep_cols].copy()

        if trades.empty:
            return self._add_empty_trade_columns(out)

        trade_agg = self._aggregate_trades_for_timeseries(trades)

        merged = pd.merge_asof(
            out.sort_values("event_time"),
            trade_agg[["event_time", "cum_position", "cum_cash"]].sort_values("event_time"),
            on="event_time",
            direction="backward",
        )

        merged["position"] = merged["cum_position"].fillna(0.0)
        merged["cash"] = merged["cum_cash"].fillna(0.0)
        merged["total_pnl_reconstructed"] = (
            merged["cash"] + merged["position"] * merged["mid_price_calc"]
        )

        merged = merged.merge(
            trade_agg[[
                "event_time", "trade_qty", "trade_price", "n_trades",
                "cash_delta", "buy_qty", "sell_qty",
            ]],
            on="event_time",
            how="left",
        )

        merged["trade_qty"] = merged["trade_qty"].fillna(0.0)
        merged["cash_delta"] = merged["cash_delta"].fillna(0.0)
        merged["n_trades"] = merged["n_trades"].fillna(0).astype(int)
        merged["buy_qty"] = merged["buy_qty"].fillna(0.0)
        merged["sell_qty"] = merged["sell_qty"].fillna(0.0)

        realized_list = []
        unrealized_list = []
        avg_cost_list = []

        trade_rows = trades.sort_values("event_time", kind="stable").reset_index(drop=True)
        seen = []
        ptr = 0

        for row in merged.itertuples(index=False):
            while ptr < len(trade_rows) and trade_rows.loc[ptr, "event_time"] <= row.event_time:
                seen.append(trade_rows.loc[ptr])
                ptr += 1

            if seen:
                seen_df = pd.DataFrame(seen)
                state = self._realized_unrealized_from_trades(seen_df, row.mid_price_calc)
                realized_list.append(state["realized_pnl"])
                unrealized_list.append(state["unrealized_pnl"])
                avg_cost_list.append(state["avg_cost"])
            else:
                realized_list.append(0.0)
                unrealized_list.append(0.0)
                avg_cost_list.append(np.nan)

        merged["avg_cost"] = avg_cost_list
        merged["realized_pnl"] = realized_list
        merged["unrealized_pnl"] = unrealized_list

        merged["trade_edge_vs_mid"] = np.where(
            merged["trade_qty"] != 0,
            np.vectorize(self._trade_edge_vs_mid)(
                merged["trade_qty"],
                merged["trade_price"],
                merged["mid_price_calc"],
            ),
            np.nan,
        )

        if "adjusted_fair" in merged.columns:
            merged["trade_edge_vs_fair"] = np.where(
                merged["trade_qty"] != 0,
                np.vectorize(self._trade_edge_vs_fair)(
                    merged["trade_qty"],
                    merged["trade_price"],
                    merged["adjusted_fair"],
                ),
                np.nan,
            )
        else:
            merged["trade_edge_vs_fair"] = np.nan

        merged["position_change"] = merged["position"].diff().fillna(merged["position"])
        merged["pnl_change"] = merged["total_pnl_reconstructed"].diff().fillna(0.0)
        merged["drawdown"] = (
            merged["total_pnl_reconstructed"]
            - merged["total_pnl_reconstructed"].cummax()
        )
        merged["unrealized_peak"] = merged["unrealized_pnl"].cummax()
        merged["unrealized_giveback"] = merged["unrealized_peak"] - merged["unrealized_pnl"]

        return merged

    def _add_empty_trade_columns(self, out: pd.DataFrame) -> pd.DataFrame:
        out = out.copy()
        out["trade_qty"] = 0.0
        out["trade_price"] = np.nan
        out["n_trades"] = 0
        out["cash_delta"] = 0.0
        out["position"] = 0.0
        out["cash"] = 0.0
        out["avg_cost"] = np.nan
        out["realized_pnl"] = 0.0
        out["unrealized_pnl"] = 0.0
        out["total_pnl_reconstructed"] = 0.0
        out["trade_edge_vs_mid"] = np.nan
        out["trade_edge_vs_fair"] = np.nan
        out["position_change"] = 0.0
        out["pnl_change"] = 0.0
        out["drawdown"] = 0.0
        out["unrealized_peak"] = 0.0
        out["unrealized_giveback"] = 0.0
        return out

    def _aggregate_trades_for_timeseries(self, trades: pd.DataFrame) -> pd.DataFrame:
        tmp = trades.copy()
        tmp["px_absqty"] = tmp["price"] * tmp["abs_qty"]
        tmp["buy_qty_component"] = np.where(tmp["signed_qty"] > 0, tmp["signed_qty"], 0.0)
        tmp["sell_qty_component"] = np.where(tmp["signed_qty"] < 0, -tmp["signed_qty"], 0.0)

        trade_agg = (
            tmp.groupby(["event_time", "day", "timestamp"], as_index=False)
            .agg(
                trade_qty=("signed_qty", "sum"),
                cash_delta=("cash_flow", "sum"),
                abs_qty=("abs_qty", "sum"),
                px_absqty=("px_absqty", "sum"),
                n_trades=("price", "count"),
                buy_qty=("buy_qty_component", "sum"),
                sell_qty=("sell_qty_component", "sum"),
            )
            .sort_values("event_time", kind="stable")
            .reset_index(drop=True)
        )

        trade_agg["trade_price"] = np.where(
            trade_agg["abs_qty"] > 0,
            trade_agg["px_absqty"] / trade_agg["abs_qty"],
            np.nan,
        )

        trade_agg["cum_position"] = trade_agg["trade_qty"].cumsum()
        trade_agg["cum_cash"] = trade_agg["cash_delta"].cumsum()

        return trade_agg

    # ------------------------------------------------------------------
    # Summary / metrics
    # ------------------------------------------------------------------

    def summarize_position(self, product: str) -> PositionSummary:
        ts = self.build_position_timeseries(product)
        trades = self.get_submission_trades(product)

        max_abs_position = float(ts["position"].abs().max())
        abs_pos = ts["position"].abs()

        reach25 = self._first_reach_index(abs_pos, 0.25 * max_abs_position) if max_abs_position > 0 else None
        reach50 = self._first_reach_index(abs_pos, 0.50 * max_abs_position) if max_abs_position > 0 else None
        reach75 = self._first_reach_index(abs_pos, 0.75 * max_abs_position) if max_abs_position > 0 else None
        reach90 = self._first_reach_index(abs_pos, 0.90 * max_abs_position) if max_abs_position > 0 else None

        buys = ts[ts["trade_qty"] > 0]
        sells = ts[ts["trade_qty"] < 0]
        trade_rows = ts[ts["trade_qty"] != 0]

        avg_buy_edge_mid = float(buys["trade_edge_vs_mid"].mean()) if not buys.empty else None
        avg_sell_edge_mid = float(sells["trade_edge_vs_mid"].mean()) if not sells.empty else None
        avg_trade_edge_mid = float(trade_rows["trade_edge_vs_mid"].mean()) if not trade_rows.empty else None

        avg_buy_edge_fair = (
            float(buys["trade_edge_vs_fair"].mean())
            if not buys.empty and "trade_edge_vs_fair" in buys.columns else None
        )
        avg_sell_edge_fair = (
            float(sells["trade_edge_vs_fair"].mean())
            if not sells.empty and "trade_edge_vs_fair" in sells.columns else None
        )
        avg_trade_edge_fair = (
            float(trade_rows["trade_edge_vs_fair"].mean())
            if not trade_rows.empty and "trade_edge_vs_fair" in trade_rows.columns else None
        )

        pos_delta = ts["position"].diff().fillna(ts["position"])
        largest_rise = float(pos_delta.max())
        largest_drop = float(pos_delta.min())

        final_pnl = float(ts["total_pnl_reconstructed"].iloc[-1])
        avg_abs_pos = float(abs_pos.mean())

        turnover = float(trades["signed_qty"].abs().sum()) if not trades.empty else 0.0
        pnl_per_turnover = final_pnl / turnover if turnover > 0 else 0.0

        inventory_efficiency = final_pnl / max(avg_abs_pos, 1e-9) if avg_abs_pos > 0 else 0.0

        largest_buy = (
            float(trades.loc[trades["signed_qty"] > 0, "signed_qty"].max())
            if not trades.empty and (trades["signed_qty"] > 0).any()
            else 0.0
        )

        largest_sell = (
            float((-trades.loc[trades["signed_qty"] < 0, "signed_qty"]).max())
            if not trades.empty and (trades["signed_qty"] < 0).any()
            else 0.0
        )

        return PositionSummary(
            product=product,
            product_group=self.product_group(product),
            n_market_rows=len(ts),
            n_submission_trades=len(trades),
            final_position=float(ts["position"].iloc[-1]),
            final_cash=float(ts["cash"].iloc[-1]),
            final_mid=float(ts["mid_price_calc"].iloc[-1]),
            final_total_pnl=final_pnl,
            final_realized_pnl=float(ts["realized_pnl"].iloc[-1]),
            final_unrealized_pnl=float(ts["unrealized_pnl"].iloc[-1]),
            max_abs_position=max_abs_position,
            avg_abs_position=avg_abs_pos,
            turnover=turnover,
            pnl_per_turnover=pnl_per_turnover,
            max_drawdown=self._max_drawdown(ts["total_pnl_reconstructed"]),
            pnl_volatility=float(ts["pnl_change"].std(ddof=0)),
            inventory_efficiency=inventory_efficiency,
            first_reach_25pct_idx=reach25,
            first_reach_50pct_idx=reach50,
            first_reach_75pct_idx=reach75,
            first_reach_90pct_idx=reach90,
            avg_buy_edge_vs_mid=avg_buy_edge_mid,
            avg_sell_edge_vs_mid=avg_sell_edge_mid,
            avg_trade_edge_vs_mid=avg_trade_edge_mid,
            avg_buy_edge_vs_fair=avg_buy_edge_fair,
            avg_sell_edge_vs_fair=avg_sell_edge_fair,
            avg_trade_edge_vs_fair=avg_trade_edge_fair,
            peak_unrealized_pnl=float(ts["unrealized_pnl"].max()),
            final_unrealized_giveback=float(ts["unrealized_giveback"].iloc[-1]),
            largest_single_buy_qty=largest_buy,
            largest_single_sell_qty=largest_sell,
            largest_position_drop=largest_drop,
            largest_position_rise=largest_rise,
        )

    def all_position_summaries(self, products: Optional[List[str]] = None) -> pd.DataFrame:
        rows = []
        for product in products or self.products:
            if self.get_market(product).empty:
                continue
            rows.append(asdict(self.summarize_position(product)))

        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows)
        return out.sort_values("final_total_pnl", ascending=False).reset_index(drop=True)

    def metrics_table(self, product: str) -> pd.DataFrame:
        summary = self.summarize_position(product)
        rows = [{"metric": k, "value": v} for k, v in asdict(summary).items()]
        return pd.DataFrame(rows)

    def summarize_option_signal(self, product: str) -> Optional[OptionSignalSummary]:
        if product not in VOUCHER_STRIKES:
            return None

        df = self.get_option_signal_market(product)
        if df.empty or "theo" not in df.columns:
            return None

        h10 = self._premium_predictive_ic(df, 10)
        h100 = self._premium_predictive_ic(df, 100)

        return OptionSignalSummary(
            product=product,
            strike=VOUCHER_STRIKES[product],
            n_rows=len(df),
            final_mid=float(df["mid_price_calc"].iloc[-1]),
            final_theo=float(df["theo"].iloc[-1]),
            final_adjusted_fair=float(df["adjusted_fair"].iloc[-1]),
            final_theo_diff=float(df["theo_diff"].iloc[-1]),
            final_mean_theo_diff=float(df["mean_theo_diff"].iloc[-1]),
            final_residual=float(df["theo_residual"].iloc[-1]),
            avg_mid=float(df["mid_price_calc"].mean()),
            avg_theo=float(df["theo"].mean()),
            avg_adjusted_fair=float(df["adjusted_fair"].mean()),
            avg_theo_diff=float(df["theo_diff"].mean()),
            avg_abs_residual=float(df["theo_residual"].abs().mean()),
            avg_absdev_ema=float(df["residual_absdev_ema"].mean()),
            avg_delta=float(df["delta"].mean()),
            avg_vega=float(df["vega"].mean()),
            avg_spread=float(df["spread"].mean()),
            buy_signal_rows_thr_1=int((df["buy_edge_vs_fair_at_ask"] >= 1.0).sum()),
            sell_signal_rows_thr_1=int((df["sell_edge_vs_fair_at_bid"] >= 1.0).sum()),
            premium_predicts_h10_ic=h10,
            premium_predicts_h100_ic=h100,
        )

    def option_signal_table(self, products: Optional[List[str]] = None) -> pd.DataFrame:
        rows = []
        for product in products or list(VOUCHER_STRIKES):
            summary = self.summarize_option_signal(product)
            if summary is not None:
                rows.append(asdict(summary))

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)

    def _premium_predictive_ic(self, df: pd.DataFrame, horizon_rows: int) -> Optional[float]:
        if "premium_intrinsic_residual" not in df.columns:
            return None

        x = df["premium_intrinsic_residual"]
        y = df["mid_price_calc"].shift(-horizon_rows) - df["mid_price_calc"]

        valid = x.notna() & y.notna()
        if valid.sum() < 30 or x[valid].std() == 0 or y[valid].std() == 0:
            return None

        return float(x[valid].corr(y[valid]))

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def build_portfolio_timeseries(self, products: Optional[List[str]] = None) -> pd.DataFrame:
        series = []

        for product in products or self.products:
            if self.get_market(product).empty:
                continue

            ts = self.build_position_timeseries(product)
            s = ts[["event_time", "total_pnl_reconstructed", "realized_pnl", "unrealized_pnl"]].copy()
            s = s.rename(columns={
                "total_pnl_reconstructed": f"{product}_total",
                "realized_pnl": f"{product}_realized",
                "unrealized_pnl": f"{product}_unrealized",
            })
            series.append(s)

        if not series:
            return pd.DataFrame()

        out = series[0]
        for s in series[1:]:
            out = out.merge(s, on="event_time", how="outer")

        out = out.sort_values("event_time").reset_index(drop=True)
        pnl_cols = [c for c in out.columns if c != "event_time"]

        out[pnl_cols] = out[pnl_cols].ffill().fillna(0.0)

        total_cols = [c for c in out.columns if c.endswith("_total")]
        realized_cols = [c for c in out.columns if c.endswith("_realized")]
        unrealized_cols = [c for c in out.columns if c.endswith("_unrealized")]

        out["portfolio_total_pnl"] = out[total_cols].sum(axis=1)
        out["portfolio_realized_pnl"] = out[realized_cols].sum(axis=1)
        out["portfolio_unrealized_pnl"] = out[unrealized_cols].sum(axis=1)
        out["portfolio_pnl_change"] = out["portfolio_total_pnl"].diff().fillna(0.0)
        out["portfolio_drawdown"] = (
            out["portfolio_total_pnl"] - out["portfolio_total_pnl"].cummax()
        )

        return out

    def print_portfolio_summary(self, products: Optional[List[str]] = None) -> None:
        pf = self.build_portfolio_timeseries(products)
        if pf.empty:
            print("No portfolio timeseries available.")
            return

        print("\n" + "=" * 120)
        print("PORTFOLIO SUMMARY")
        print("=" * 120)
        print(f"Final total PnL:       {pf['portfolio_total_pnl'].iloc[-1]:,.2f}")
        print(f"Final realized PnL:    {pf['portfolio_realized_pnl'].iloc[-1]:,.2f}")
        print(f"Final unrealized PnL:  {pf['portfolio_unrealized_pnl'].iloc[-1]:,.2f}")
        print(f"Max drawdown:          {pf['portfolio_drawdown'].min():,.2f}")
        print(f"PnL volatility:        {pf['portfolio_pnl_change'].std(ddof=0):,.4f}")

    # ------------------------------------------------------------------
    # Trade tables / diagnostics
    # ------------------------------------------------------------------

    def print_trade_table(self, product: str, n: Optional[int] = None) -> None:
        trades = self.get_submission_trades(product)
        if trades.empty:
            print(f"No submission trades for {product}")
            return

        out = trades[["day", "timestamp", "price", "quantity", "signed_qty", "cash_flow"]].copy()
        out["cum_position"] = out["signed_qty"].cumsum()
        out["cum_cash"] = out["cash_flow"].cumsum()

        if n is not None:
            out = out.head(n)

        print(f"\n===== {product} submission trades =====")
        print(out.to_string(index=False))

    def print_strategy_diagnostics(self, product: str) -> None:
        summary = self.summarize_position(product)
        print(f"\n===== {product} diagnostics =====")
        for k, v in asdict(summary).items():
            print(f"{k}: {v}")

        if product in VOUCHER_STRIKES:
            opt = self.summarize_option_signal(product)
            if opt is not None:
                print(f"\n===== {product} option signal diagnostics =====")
                for k, v in asdict(opt).items():
                    print(f"{k}: {v}")

    # ------------------------------------------------------------------
    # Stress / parameter tests
    # ------------------------------------------------------------------

    def shock_test(self, product: str, shocks: Optional[List[float]] = None) -> pd.DataFrame:
        if shocks is None:
            # Wider shocks for high-value products, smaller products still use absolute ticks.
            shocks = [-1.0, -2.0, -5.0, -10.0, -20.0, -50.0]

        ts = self.build_position_timeseries(product)
        rows = []

        for shock in shocks:
            shocked_pnl = ts["cash"] + ts["position"] * (ts["mid_price_calc"] + shock)
            rows.append({
                "product": product,
                "shock": shock,
                "final_pnl_under_shock": float(shocked_pnl.iloc[-1]),
                "worst_pnl_under_shock": float(shocked_pnl.min()),
                "final_position": float(ts["position"].iloc[-1]),
            })

        return pd.DataFrame(rows)

    def signal_threshold_sweep(
        self,
        product: str,
        thresholds: Optional[List[float]] = None,
        horizon_rows: int = 100,
    ) -> pd.DataFrame:
        """
        Parameter-improvement helper for voucher fair-value thresholds.

        For each threshold:
          buy signal  = adjusted_fair - ask >= threshold
          sell signal = bid - adjusted_fair >= threshold

        Future edge estimate:
          buy  -> future_mid - ask
          sell -> bid - future_mid

        Positive future_edge means the signal direction was profitable before
        inventory, queueing, and spread/friction complications.
        """
        if thresholds is None:
            thresholds = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]

        if product not in VOUCHER_STRIKES:
            return pd.DataFrame()

        df = self.get_option_signal_market(product)

        if df.empty or "adjusted_fair" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["future_mid"] = df["mid_price_calc"].shift(-horizon_rows)
        df["buy_future_edge"] = df["future_mid"] - df["ask_price_1"]
        df["sell_future_edge"] = df["bid_price_1"] - df["future_mid"]

        rows = []

        for thr in thresholds:
            buy = df[df["buy_edge_vs_fair_at_ask"] >= thr]
            sell = df[df["sell_edge_vs_fair_at_bid"] >= thr]

            both_edges = pd.concat([
                buy["buy_future_edge"].dropna(),
                sell["sell_future_edge"].dropna(),
            ])

            rows.append({
                "product": product,
                "horizon_rows": horizon_rows,
                "threshold": thr,

                "n_buy_signals": int(len(buy)),
                "buy_hit_rate": float((buy["buy_future_edge"] > 0).mean()) if len(buy) else np.nan,
                "buy_mean_future_edge": float(buy["buy_future_edge"].mean()) if len(buy) else np.nan,
                "buy_median_future_edge": float(buy["buy_future_edge"].median()) if len(buy) else np.nan,

                "n_sell_signals": int(len(sell)),
                "sell_hit_rate": float((sell["sell_future_edge"] > 0).mean()) if len(sell) else np.nan,
                "sell_mean_future_edge": float(sell["sell_future_edge"].mean()) if len(sell) else np.nan,
                "sell_median_future_edge": float(sell["sell_future_edge"].median()) if len(sell) else np.nan,

                "n_total_signals": int(len(buy) + len(sell)),
                "combined_hit_rate": float((both_edges > 0).mean()) if len(both_edges) else np.nan,
                "combined_mean_future_edge": float(both_edges.mean()) if len(both_edges) else np.nan,
                "combined_median_future_edge": float(both_edges.median()) if len(both_edges) else np.nan,
            })

        return pd.DataFrame(rows)

    def print_threshold_sweeps(
        self,
        products: Optional[List[str]] = None,
        horizons: Iterable[int] = (10, 100),
    ) -> None:
        products = products or ["VEV_5300", "VEV_5400", "VEV_5500"]

        for product in products:
            if self.get_market(product).empty:
                continue

            for h in horizons:
                table = self.signal_threshold_sweep(product, horizon_rows=h)
                if table.empty:
                    continue

                print("\n" + "=" * 120)
                print(f"{product} THRESHOLD SWEEP | horizon_rows={h}")
                print("=" * 120)

                cols = [
                    "threshold",
                    "n_total_signals",
                    "combined_hit_rate",
                    "combined_mean_future_edge",
                    "n_buy_signals",
                    "buy_hit_rate",
                    "buy_mean_future_edge",
                    "n_sell_signals",
                    "sell_hit_rate",
                    "sell_mean_future_edge",
                ]
                print(table[cols].to_string(index=False))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_portfolio_summary(self, products: Optional[List[str]] = None) -> None:
        pf = self.build_portfolio_timeseries(products)
        if pf.empty:
            print("No portfolio timeseries available.")
            return

        fig, axs = plt.subplots(3, 1, figsize=(17, 10), sharex=True)

        axs[0].plot(pf["event_time"], pf["portfolio_total_pnl"], label="total")
        axs[0].plot(pf["event_time"], pf["portfolio_realized_pnl"], label="realized")
        axs[0].plot(pf["event_time"], pf["portfolio_unrealized_pnl"], label="unrealized")
        axs[0].axhline(0, linestyle=":")
        axs[0].set_title("Portfolio PnL")
        axs[0].legend()

        axs[1].plot(pf["event_time"], pf["portfolio_drawdown"], label="drawdown")
        axs[1].axhline(0, linestyle=":")
        axs[1].set_title("Portfolio drawdown")
        axs[1].legend()

        total_cols = [c for c in pf.columns if c.endswith("_total")]
        for col in total_cols:
            product = col[:-6]
            axs[2].plot(pf["event_time"], pf[col], label=product)

        axs[2].axhline(0, linestyle=":")
        axs[2].set_title("Product contribution")
        axs[2].legend(ncol=3, fontsize=8)

        axs[2].set_xlabel("event_time")
        plt.tight_layout()
        plt.show()

    def plot_position_summary(self, product: str, show_site_pnl: bool = True) -> None:
        ts = self.build_position_timeseries(product)
        summary = self.summarize_position(product)

        fig, axs = plt.subplots(6, 1, figsize=(17, 18), sharex=True)

        axs[0].plot(ts["row_id"], ts["mid_price_calc"], label="mid")
        axs[0].plot(ts["row_id"], ts["bid_price_1"], alpha=0.5, label="bid1")
        axs[0].plot(ts["row_id"], ts["ask_price_1"], alpha=0.5, label="ask1")

        trade_points = ts[ts["trade_qty"] != 0].copy()
        if not trade_points.empty:
            buys = trade_points[trade_points["trade_qty"] > 0]
            sells = trade_points[trade_points["trade_qty"] < 0]

            if not buys.empty:
                axs[0].scatter(
                    buys["row_id"], buys["trade_price"],
                    marker="^", s=45, label="our buys"
                )
            if not sells.empty:
                axs[0].scatter(
                    sells["row_id"], sells["trade_price"],
                    marker="v", s=45, label="our sells"
                )

        axs[0].set_title(
            f"{product} price + actual submission fills | "
            f"final pnl={summary.final_total_pnl:.1f} | final pos={summary.final_position:.1f}"
        )
        axs[0].legend()

        axs[1].step(ts["row_id"], ts["position"], where="post", label="position")
        axs[1].axhline(0, linestyle=":")
        axs[1].set_title(
            f"Position | max_abs={summary.max_abs_position:.1f} | avg_abs={summary.avg_abs_position:.1f}"
        )
        axs[1].legend()

        axs[2].plot(ts["row_id"], ts["realized_pnl"], label="realized pnl")
        axs[2].plot(ts["row_id"], ts["unrealized_pnl"], label="unrealized pnl")
        axs[2].plot(ts["row_id"], ts["total_pnl_reconstructed"], label="reconstructed total pnl")
        if show_site_pnl and "profit_and_loss" in ts.columns:
            axs[2].plot(ts["row_id"], ts["profit_and_loss"], alpha=0.7, label="site profit_and_loss")
        axs[2].axhline(0, linestyle=":")
        axs[2].set_title(
            f"PnL decomposition | max_dd={summary.max_drawdown:.1f} | pnl_vol={summary.pnl_volatility:.2f}"
        )
        axs[2].legend()

        axs[3].plot(ts["row_id"], ts["drawdown"], label="drawdown")
        axs[3].plot(ts["row_id"], ts["unrealized_giveback"], label="unrealized giveback", alpha=0.85)
        axs[3].axhline(0, linestyle=":")
        axs[3].set_title(
            f"Risk view | peak_unrl={summary.peak_unrealized_pnl:.1f} | "
            f"final giveback={summary.final_unrealized_giveback:.1f}"
        )
        axs[3].legend()

        axs[4].bar(ts["row_id"], ts["trade_qty"].fillna(0.0), width=1.0, label="trade qty")
        axs[4].axhline(0, linestyle=":")
        axs[4].set_title(
            f"Trade flow | turnover={summary.turnover:.1f} | "
            f"largest buy={summary.largest_single_buy_qty:.1f} | largest sell={summary.largest_single_sell_qty:.1f}"
        )
        axs[4].legend()

        axs[5].plot(ts["row_id"], ts["spread"], label="spread")
        axs[5].plot(ts["row_id"], ts["microprice_offset"], label="microprice_offset", alpha=0.8)
        axs[5].axhline(0, linestyle=":")
        axs[5].set_title(
            f"Microstructure | buy_edge_mid={summary.avg_buy_edge_vs_mid} | sell_edge_mid={summary.avg_sell_edge_vs_mid}"
        )
        axs[5].legend()

        self._annotate_logs(product, ts, axs)

        axs[5].set_xlabel("market row index")
        plt.tight_layout()
        plt.show()

    def plot_option_diagnostics(self, product: str) -> None:
        if product not in VOUCHER_STRIKES:
            print(f"{product} is not a configured voucher.")
            return

        ts = self.build_position_timeseries(product)
        if "theo" not in ts.columns:
            print(f"No option diagnostics for {product}")
            return

        summary = self.summarize_position(product)

        fig, axs = plt.subplots(6, 1, figsize=(17, 18), sharex=True)

        trade_points = ts[ts["trade_qty"] != 0].copy()

        axs[0].plot(ts["row_id"], ts["mid_price_calc"], label="voucher mid")
        axs[0].plot(ts["row_id"], ts["theo"], label="BS theo", alpha=0.85)
        axs[0].plot(ts["row_id"], ts["adjusted_fair"], label="adjusted fair", alpha=0.85)
        axs[0].plot(ts["row_id"], ts["intrinsic"], label="intrinsic", alpha=0.65)
        if not trade_points.empty:
            buys = trade_points[trade_points["trade_qty"] > 0]
            sells = trade_points[trade_points["trade_qty"] < 0]
            if not buys.empty:
                axs[0].scatter(buys["row_id"], buys["trade_price"], marker="^", s=45, label="buys")
            if not sells.empty:
                axs[0].scatter(sells["row_id"], sells["trade_price"], marker="v", s=45, label="sells")
        axs[0].set_title(f"{product} valuation | final pnl={summary.final_total_pnl:.1f}")
        axs[0].legend()

        axs[1].plot(ts["row_id"], ts["theo_diff"], label="market mid - theo")
        axs[1].plot(ts["row_id"], ts["mean_theo_diff"], label="EMA theo diff")
        axs[1].plot(ts["row_id"], ts["theo_residual"], label="residual")
        axs[1].axhline(0, linestyle=":")
        axs[1].set_title("Theoretical-diff normalization")
        axs[1].legend()

        axs[2].plot(ts["row_id"], ts["buy_edge_vs_fair_at_ask"], label="buy edge at ask")
        axs[2].plot(ts["row_id"], ts["sell_edge_vs_fair_at_bid"], label="sell edge at bid")
        axs[2].axhline(0, linestyle=":")
        axs[2].axhline(1.0, linestyle="--", alpha=0.5, label="threshold 1")
        axs[2].set_title(
            f"Executable edges | avg trade edge fair={summary.avg_trade_edge_vs_fair}"
        )
        axs[2].legend()

        axs[3].step(ts["row_id"], ts["position"], where="post", label="position")
        axs[3].axhline(0, linestyle=":")
        axs[3].set_title("Position")
        axs[3].legend()

        axs[4].plot(ts["row_id"], ts["realized_pnl"], label="realized")
        axs[4].plot(ts["row_id"], ts["unrealized_pnl"], label="unrealized")
        axs[4].plot(ts["row_id"], ts["total_pnl_reconstructed"], label="total")
        axs[4].axhline(0, linestyle=":")
        axs[4].set_title("PnL")
        axs[4].legend()

        axs[5].plot(ts["row_id"], ts["delta"], label="delta")
        axs[5].plot(ts["row_id"], ts["vega"], label="vega")
        axs[5].plot(ts["row_id"], ts["iv"], label="iv")
        axs[5].set_title("Greeks / IV")
        axs[5].legend()

        self._annotate_logs(product, ts, axs)

        axs[5].set_xlabel("market row index")
        plt.tight_layout()
        plt.show()

    def _annotate_logs(self, product: str, ts: pd.DataFrame, axs) -> None:
        if self.logs_df is None or self.logs_df.empty:
            return

        logs = self.logs_df.copy()
        if "timestamp" not in logs.columns:
            return

        warn = logs[
            logs["sandboxLog"].str.contains(product, na=False)
            | logs["lambdaLog"].str.contains(product, na=False)
        ].copy()

        if warn.empty:
            return

        market_times = ts[["timestamp", "row_id"]].drop_duplicates("timestamp")
        warn = warn.merge(market_times, on="timestamp", how="left")

        for x in warn["row_id"].dropna().tolist():
            for ax in axs:
                ax.axvline(x, linestyle=":", alpha=0.08)

    def plot_trade_edges(self, product: str) -> None:
        ts = self.build_position_timeseries(product)
        trades = ts[ts["trade_qty"] != 0].copy()
        if trades.empty:
            print(f"No trades for {product}")
            return

        fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        axs[0].bar(trades["row_id"], trades["trade_edge_vs_mid"])
        axs[0].axhline(0, linestyle=":")
        axs[0].set_title("Trade edge vs mid, positive = good")

        if "trade_edge_vs_fair" in trades.columns and trades["trade_edge_vs_fair"].notna().any():
            axs[1].bar(trades["row_id"], trades["trade_edge_vs_fair"])
            axs[1].axhline(0, linestyle=":")
            axs[1].set_title("Trade edge vs adjusted fair, positive = good")
        else:
            axs[1].text(0.5, 0.5, "No fair-value edge for this product", ha="center", va="center")
            axs[1].set_title("Trade edge vs adjusted fair")

        axs[2].scatter(
            trades["row_id"],
            trades["trade_price"],
            c=np.where(trades["trade_qty"] > 0, 1, -1),
            s=45,
        )
        axs[2].plot(ts["row_id"], ts["mid_price_calc"], alpha=0.7, label="mid")
        if "adjusted_fair" in ts.columns:
            axs[2].plot(ts["row_id"], ts["adjusted_fair"], alpha=0.7, label="adjusted fair")
        axs[2].set_title("Trade prices vs mid/fair")
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    def plot_threshold_sweep(
        self,
        product: str,
        horizon_rows: int = 100,
    ) -> None:
        table = self.signal_threshold_sweep(product, horizon_rows=horizon_rows)
        if table.empty:
            print(f"No threshold sweep available for {product}.")
            return

        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axs[0].plot(table["threshold"], table["combined_mean_future_edge"], marker="o", label="combined")
        axs[0].plot(table["threshold"], table["buy_mean_future_edge"], marker="o", label="buy")
        axs[0].plot(table["threshold"], table["sell_mean_future_edge"], marker="o", label="sell")
        axs[0].axhline(0, linestyle=":")
        axs[0].set_title(f"{product} threshold sweep, future edge, h={horizon_rows}")
        axs[0].legend()

        axs[1].plot(table["threshold"], table["n_total_signals"], marker="o", label="n total")
        axs[1].plot(table["threshold"], table["n_buy_signals"], marker="o", label="n buy")
        axs[1].plot(table["threshold"], table["n_sell_signals"], marker="o", label="n sell")
        axs[1].set_title("Signal count")
        axs[1].legend()

        axs[1].set_xlabel("threshold")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Convenience reports
    # ------------------------------------------------------------------

    def print_all_summaries(self, products: Optional[List[str]] = None) -> pd.DataFrame:
        summary = self.all_position_summaries(products)

        if summary.empty:
            print("No summaries available.")
            return summary

        pd.set_option("display.width", 220)
        pd.set_option("display.max_columns", 100)
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

        print("\n" + "=" * 120)
        print("PRODUCT PNL / POSITION SUMMARY")
        print("=" * 120)

        cols = [
            "product", "product_group", "n_submission_trades",
            "final_total_pnl", "final_realized_pnl", "final_unrealized_pnl",
            "final_position", "turnover", "pnl_per_turnover",
            "max_abs_position", "avg_abs_position", "max_drawdown",
            "avg_trade_edge_vs_mid", "avg_trade_edge_vs_fair",
        ]

        print(summary[cols].to_string(index=False))

        return summary

    def print_option_signal_table(self, products: Optional[List[str]] = None) -> pd.DataFrame:
        table = self.option_signal_table(products)

        if table.empty:
            print("No option signal table available.")
            return table

        pd.set_option("display.width", 220)
        pd.set_option("display.max_columns", 100)
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

        print("\n" + "=" * 120)
        print("OPTION / VOUCHER SIGNAL SUMMARY")
        print("=" * 120)

        cols = [
            "product", "strike",
            "avg_mid", "avg_theo", "avg_adjusted_fair",
            "avg_theo_diff", "avg_abs_residual",
            "avg_delta", "avg_vega", "avg_spread",
            "buy_signal_rows_thr_1", "sell_signal_rows_thr_1",
            "premium_predicts_h10_ic", "premium_predicts_h100_ic",
        ]

        print(table[cols].to_string(index=False))

        return table

    def plot_shock_table(self, product: str) -> None:
        shock_df = self.shock_test(product)
        print(f"\n===== {product} shock test =====")
        print(shock_df.to_string(index=False))

    def run_full_report(self, product: str, n_trade_rows: int = 30) -> None:
        summary = self.summarize_position(product)

        print("\n===== SUMMARY =====")
        print(summary)

        self.print_trade_table(product, n=n_trade_rows)

        print(f"\n===== {product} metrics table =====")
        print(self.metrics_table(product).to_string(index=False))

        self.plot_shock_table(product)

        if product in VOUCHER_STRIKES:
            self.plot_option_diagnostics(product)
            self.plot_threshold_sweep(product, horizon_rows=100)
        else:
            self.plot_position_summary(product, show_site_pnl=True)

        self.plot_trade_edges(product)

    def run_group_report(
        self,
        products: Optional[List[str]] = None,
        detailed_products: Optional[List[str]] = None,
        n_trade_rows: int = 30,
    ) -> None:
        products = products or self.products

        self.print_portfolio_summary(products)
        self.print_all_summaries(products)
        self.print_option_signal_table([p for p in products if p in VOUCHER_STRIKES])
        self.print_threshold_sweeps(
            products=[p for p in ["VEV_5300", "VEV_5400", "VEV_5500"] if p in products],
            horizons=(10, 100),
        )

        self.plot_portfolio_summary(products)

        if detailed_products is None:
            # Focus on products most likely to matter from previous diagnostics.
            detailed_products = [
                UNDERLYING,
                "VEV_5300",
                "VEV_5400",
                "VEV_5500",
                HYDROGEL,
            ]

        for product in detailed_products:
            if product in products and not self.get_market(product).empty:
                self.run_full_report(product, n_trade_rows=n_trade_rows)


if __name__ == "__main__":
    LOG_PATH = "/Users/janekczajnik/Downloads/372318 2/372318.log"

    analyzer = Round3LogAnalyzer(LOG_PATH, products=ROUND3_PRODUCTS)
    analyzer.load_all()

    analyzer.run_group_report(
        products=ROUND3_PRODUCTS,
        detailed_products=[
            "VELVETFRUIT_EXTRACT",
            "VEV_5300",
            "VEV_5400",
            "VEV_5500",
            "HYDROGEL_PACK",
        ],
        n_trade_rows=40,
    )
