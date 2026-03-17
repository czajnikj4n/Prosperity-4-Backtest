import json
import io
import pandas as pd
import matplotlib.pyplot as plt


class Backtester:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.data = None
        self.activities_df = None
        self.trades_df = None

    # -------------------------
    # LOAD LOG FILE
    # -------------------------
    def load_log(self):
        with open(self.log_file_path, "r") as f:
            content = f.read()

        try:
            self.data = json.loads(content)
        except json.JSONDecodeError:
            self.data = json.loads(content.splitlines()[0])

    # -------------------------
    # PARSE ACTIVITIES
    # -------------------------
    def parse_activities(self):
        raw_csv = self.data.get("activitiesLog", "")
        csv_file = io.StringIO(raw_csv)
        self.activities_df = pd.read_csv(csv_file, sep=";")

    # -------------------------
    # PARSE TRADES
    # -------------------------
    def parse_trades(self):
        trades = self.data.get("tradeHistory", [])
        self.trades_df = pd.DataFrame(trades)

    # -------------------------
    # HELPERS
    # -------------------------
    @staticmethod
    def _signed_qty_from_row(row):
        buyer = row.get("buyer", "")
        seller = row.get("seller", "")

        if buyer == "SUBMISSION":
            return row["quantity"]
        if seller == "SUBMISSION":
            return -row["quantity"]
        return 0

    @staticmethod
    def _true_edge_from_row(row):
        signed_qty = row["signed_qty"]
        price = row["price"]
        mid = row["mid_price_calc"]

        if signed_qty > 0:
            return mid - price
        if signed_qty < 0:
            return price - mid
        return 0.0

    # -------------------------
    # PREPROCESS
    # -------------------------
    def preprocess(self):
        if self.activities_df is None or self.activities_df.empty:
            raise ValueError("activities_df is empty. Did you call parse_activities()?")

        df = self.activities_df.copy()

        numeric_cols = [
            "day",
            "timestamp",
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

        # Core derived columns
        df["mid_price_calc"] = (df["bid_price_1"] + df["ask_price_1"]) / 2
        df["spread"] = df["ask_price_1"] - df["bid_price_1"]

        # Rolling volatility on mid changes
        df["mid_return"] = df.groupby("product")["mid_price_calc"].diff()
        df["rolling_vol_50"] = (
            df.groupby("product")["mid_return"]
            .rolling(50)
            .std()
            .reset_index(level=0, drop=True)
        )

        # Order book imbalance
        df["imbalance_1"] = (
                (df["bid_volume_1"] - df["ask_volume_1"]) /
                (df["bid_volume_1"] + df["ask_volume_1"] + 1e-9)
        )

        df["bid_depth_3"] = (
                df["bid_volume_1"].fillna(0)
                + df["bid_volume_2"].fillna(0)
                + df["bid_volume_3"].fillna(0)
        )

        df["ask_depth_3"] = (
                df["ask_volume_1"].fillna(0)
                + df["ask_volume_2"].fillna(0)
                + df["ask_volume_3"].fillna(0)
        )

        df["imbalance_3"] = (
                (df["bid_depth_3"] - df["ask_depth_3"]) /
                (df["bid_depth_3"] + df["ask_depth_3"] + 1e-9)
        )

        df["front_share_bid"] = df["bid_volume_1"] / (df["bid_depth_3"] + 1e-9)
        df["front_share_ask"] = df["ask_volume_1"] / (df["ask_depth_3"] + 1e-9)

        df["depth_slope_bid"] = df["bid_volume_1"].fillna(0) - df["bid_volume_3"].fillna(0)
        df["depth_slope_ask"] = df["ask_volume_1"].fillna(0) - df["ask_volume_3"].fillna(0)

        df["bid_gap_12"] = df["bid_price_1"] - df["bid_price_2"]
        df["bid_gap_23"] = df["bid_price_2"] - df["bid_price_3"]
        df["ask_gap_12"] = df["ask_price_2"] - df["ask_price_1"]
        df["ask_gap_23"] = df["ask_price_3"] - df["ask_price_2"]

        df["microprice"] = (
                                   df["ask_price_1"] * df["bid_volume_1"] +
                                   df["bid_price_1"] * df["ask_volume_1"]
                           ) / (df["bid_volume_1"] + df["ask_volume_1"] + 1e-9)

        df["microprice_offset"] = df["microprice"] - df["mid_price_calc"]

        bid12 = df["bid_volume_1"].fillna(0) + df["bid_volume_2"].fillna(0)
        ask12 = df["ask_volume_1"].fillna(0) + df["ask_volume_2"].fillna(0)
        df["imbalance_12"] = (bid12 - ask12) / (bid12 + ask12 + 1e-9)

        # Future mid moves / markout horizons
        for horizon in [1, 3, 5, 10]:
            df[f"future_mid_{horizon}"] = (
                df.groupby("product")["mid_price_calc"].shift(-horizon)
            )
            df[f"future_move_{horizon}"] = (
                    df[f"future_mid_{horizon}"] - df["mid_price_calc"]
            )

        self.activities_df = df.sort_values(["product", "timestamp"]).reset_index(drop=True)

        if self.trades_df is not None and not self.trades_df.empty:
            trades = self.trades_df.copy()
            for col in ["timestamp", "price", "quantity"]:
                if col in trades.columns:
                    trades[col] = pd.to_numeric(trades[col], errors="coerce")
            self.trades_df = trades.sort_values("timestamp").reset_index(drop=True)

    # -------------------------
    # FILTER PRODUCT
    # -------------------------
    def get_product_df(self, product: str):
        if self.activities_df is None or self.activities_df.empty:
            return pd.DataFrame()

        df = self.activities_df[self.activities_df["product"] == product].copy()
        return df.sort_values("timestamp").reset_index(drop=True)

    # -------------------------
    # INVENTORY TRACKING
    # -------------------------
    def compute_position(self, product: str):
        if self.trades_df is None or self.trades_df.empty:
            return None

        df = self.trades_df[self.trades_df["symbol"] == product].copy()
        if df.empty:
            return None

        df["signed_qty"] = df.apply(self._signed_qty_from_row, axis=1)
        df["position"] = df["signed_qty"].cumsum()
        return df[["timestamp", "price", "quantity", "signed_qty", "position"]]

    # -------------------------
    # SIGNED TRADE FLOW
    # -------------------------
    def compute_trade_flow(self, product: str):
        if self.trades_df is None or self.trades_df.empty:
            return None

        df = self.trades_df[self.trades_df["symbol"] == product].copy()
        if df.empty:
            return None

        df["signed_qty"] = df.apply(self._signed_qty_from_row, axis=1)
        df["cum_signed_flow"] = df["signed_qty"].cumsum()
        return df[["timestamp", "price", "quantity", "signed_qty", "cum_signed_flow"]]

    # -------------------------
    # TRADE STATE MERGE
    # -------------------------
    def get_trade_state_df(self, product: str):
        if self.trades_df is None or self.trades_df.empty:
            return None

        trades = self.trades_df[self.trades_df["symbol"] == product].copy()
        if trades.empty:
            return None

        market = self.get_product_df(product)
        if market.empty:
            return None

        merge_cols = [
            "timestamp",
            "mid_price_calc",
            "spread",
            "rolling_vol_50",
            "imbalance_1",
            "imbalance_12",
            "future_move_1",
            "future_move_3",
            "future_move_5",
            "future_move_10",
            "future_mid_1",
            "future_mid_3",
            "future_mid_5",
            "future_mid_10",
            "imbalance_3",
            "front_share_bid",
            "front_share_ask",
            "depth_slope_bid",
            "depth_slope_ask",
            "bid_gap_12",
            "bid_gap_23",
            "ask_gap_12",
            "ask_gap_23",
            "microprice",
            "microprice_offset",
        ]

        merged = pd.merge_asof(
            trades.sort_values("timestamp"),
            market[merge_cols].sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

        merged["signed_qty"] = merged.apply(self._signed_qty_from_row, axis=1)
        merged["true_edge"] = merged.apply(self._true_edge_from_row, axis=1)

        # Signed direction for markouts
        merged["trade_dir"] = merged["signed_qty"].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )

        for horizon in [1, 3, 5, 10]:
            merged[f"markout_{horizon}"] = (
                    (merged[f"future_mid_{horizon}"] - merged["price"]) * merged["trade_dir"]
            )

        return merged

    # -------------------------
    # BUY/SELL BY PRICE
    # -------------------------
    def plot_trade_distribution_horizontal(self, ax, product: str):
        if self.trades_df is None or self.trades_df.empty:
            ax.set_title(f"{product} Buy/Sell Quantity by Price (no trades)")
            return

        trade_df = self.trades_df[self.trades_df["symbol"] == product].copy()
        if trade_df.empty:
            ax.set_title(f"{product} Buy/Sell Quantity by Price (no trades)")
            return

        trade_df = trade_df.dropna(subset=["price", "quantity"]).copy()
        if trade_df.empty:
            ax.set_title(f"{product} Buy/Sell Quantity by Price (no prices)")
            return

        trade_df["price"] = trade_df["price"].astype(int)
        trade_df["quantity"] = pd.to_numeric(trade_df["quantity"], errors="coerce").fillna(0)
        trade_df["signed_qty"] = trade_df.apply(self._signed_qty_from_row, axis=1)

        buys = trade_df[trade_df["signed_qty"] > 0].groupby("price")["signed_qty"].sum()
        sells = -trade_df[trade_df["signed_qty"] < 0].groupby("price")["signed_qty"].sum()

        all_prices = sorted(set(buys.index).union(set(sells.index)))
        if not all_prices:
            ax.set_title(f"{product} Buy/Sell Quantity by Price (no submission trades)")
            return

        buys = buys.reindex(all_prices, fill_value=0)
        sells = sells.reindex(all_prices, fill_value=0)

        y_labels = [str(p) for p in all_prices]

        ax.barh(y_labels, buys.values, label="Buys")
        ax.barh(y_labels, -sells.values, label="Sells")
        ax.axvline(0, linestyle=":")
        ax.set_title(f"{product} Buy/Sell Quantity by Price")
        ax.set_xlabel("Buy qty  ←  0  →  Sell qty")
        ax.set_ylabel("Price")
        ax.legend()

    # -------------------------
    # 8-PANEL PRODUCT DASHBOARD
    # -------------------------
    def plot_product_dashboard(self, product: str):
        df = self.get_product_df(product)
        pos_df = self.compute_position(product)
        flow_df = self.compute_trade_flow(product)

        fig, axs = plt.subplots(4, 2, figsize=(18, 12))

        if df.empty:
            for i in range(4):
                for j in range(2):
                    axs[i, j].set_title(f"{product} - no data")
            plt.tight_layout()
            plt.show()
            return

        # Row 0
        axs[0, 0].plot(df["timestamp"], df["mid_price_calc"], label="Mid")
        axs[0, 0].plot(df["timestamp"], df["bid_price_1"], alpha=0.5, label="Bid1")
        axs[0, 0].plot(df["timestamp"], df["ask_price_1"], alpha=0.5, label="Ask1")
        axs[0, 0].set_title(f"{product} Prices")
        axs[0, 0].legend()

        axs[0, 1].plot(df["timestamp"], df["profit_and_loss"])
        axs[0, 1].set_title("PnL")

        # Row 1
        if pos_df is not None and not pos_df.empty:
            axs[1, 0].step(pos_df["timestamp"], pos_df["position"], where="post")
            axs[1, 0].axhline(80, linestyle="--")
            axs[1, 0].axhline(-80, linestyle="--")
            axs[1, 0].axhline(0, linestyle=":")
        axs[1, 0].set_title("Position")

        self.plot_trade_distribution_horizontal(axs[1, 1], product)

        # Row 2
        axs[2, 0].plot(df["timestamp"], df["rolling_vol_50"])
        axs[2, 0].set_title("Volatility")

        axs[2, 1].plot(df["timestamp"], df["spread"])
        axs[2, 1].set_title("Spread")

        # Row 3
        axs[3, 0].plot(df["timestamp"], df["bid_volume_1"], label="Bid")
        axs[3, 0].plot(df["timestamp"], df["ask_volume_1"], label="Ask")
        axs[3, 0].set_title("Top-of-book Volume")
        axs[3, 0].legend()

        if flow_df is not None and not flow_df.empty:
            axs[3, 1].step(flow_df["timestamp"], flow_df["cum_signed_flow"], where="post")
            axs[3, 1].axhline(0, linestyle=":")
        axs[3, 1].set_title("Signed Trade Flow")

        plt.tight_layout()
        plt.show()

    # -------------------------
    # QUANT DIAGNOSTICS
    # -------------------------
    def run_diagnostics(self, product: str):
        print(f"\n========== {product} DIAGNOSTICS ==========\n")

        df = self.get_product_df(product).copy()
        pos_df = self.compute_position(product)
        trade_state_df = self.get_trade_state_df(product)

        if df.empty:
            print("No data.")
            return

        df["pnl_change"] = df["profit_and_loss"].diff()

        total_pnl = df["profit_and_loss"].iloc[-1]
        avg_pnl = df["pnl_change"].mean()

        print(f"Total PnL: {total_pnl:.2f}")
        print(f"Avg PnL per step: {avg_pnl:.4f}")

        # -------------------------
        # PnL vs Volatility
        # -------------------------
        df["vol_bucket"] = pd.qcut(df["rolling_vol_50"], 4, labels=False, duplicates="drop")
        print("\nPnL vs Volatility (low → high):")
        print(df.groupby("vol_bucket")["pnl_change"].mean())

        # -------------------------
        # PnL vs existing imbalance features
        # -------------------------
        df["imb_bucket"] = pd.qcut(df["imbalance_1"], 5, labels=False, duplicates="drop")
        print("\nPnL vs Top-of-book imbalance (negative → positive):")
        print(df.groupby("imb_bucket")["pnl_change"].mean())

        df["imb12_bucket"] = pd.qcut(df["imbalance_12"], 5, labels=False, duplicates="drop")
        print("\nPnL vs 2-level imbalance (negative → positive):")
        print(df.groupby("imb12_bucket")["pnl_change"].mean())

        df["imb3_bucket"] = pd.qcut(df["imbalance_3"], 5, labels=False, duplicates="drop")
        print("\nPnL vs 3-level imbalance (negative → positive):")
        print(df.groupby("imb3_bucket")["pnl_change"].mean())

        # -------------------------
        # PnL vs microprice / depth shape
        # -------------------------
        df["micro_bucket"] = pd.qcut(df["microprice_offset"], 5, labels=False, duplicates="drop")
        print("\nPnL vs microprice offset (negative → positive):")
        print(df.groupby("micro_bucket")["pnl_change"].mean())

        df["front_bid_bucket"] = pd.qcut(df["front_share_bid"], 5, labels=False, duplicates="drop")
        df["front_ask_bucket"] = pd.qcut(df["front_share_ask"], 5, labels=False, duplicates="drop")

        print("\nPnL vs bid front-share:")
        print(df.groupby("front_bid_bucket")["pnl_change"].mean())

        print("\nPnL vs ask front-share:")
        print(df.groupby("front_ask_bucket")["pnl_change"].mean())

        df["depth_slope_bid_bucket"] = pd.qcut(df["depth_slope_bid"], 5, labels=False, duplicates="drop")
        df["depth_slope_ask_bucket"] = pd.qcut(df["depth_slope_ask"], 5, labels=False, duplicates="drop")

        print("\nPnL vs bid depth slope:")
        print(df.groupby("depth_slope_bid_bucket")["pnl_change"].mean())

        print("\nPnL vs ask depth slope:")
        print(df.groupby("depth_slope_ask_bucket")["pnl_change"].mean())

        # -------------------------
        # PnL vs Future move
        # -------------------------
        for horizon in [1, 3, 5]:
            bucket_col = f"future_bucket_{horizon}"
            move_col = f"future_move_{horizon}"
            df[bucket_col] = pd.qcut(df[move_col], 5, labels=False, duplicates="drop")
            print(f"\nPnL vs future move bucket ({horizon}) (down → up):")
            print(df.groupby(bucket_col)["pnl_change"].mean())

        # -------------------------
        # PnL vs Position
        # -------------------------
        if pos_df is not None and not pos_df.empty:
            merged = pd.merge_asof(
                df.sort_values("timestamp"),
                pos_df.sort_values("timestamp"),
                on="timestamp"
            )
            merged["pos_bucket"] = pd.cut(merged["position"], 5)

            print("\nPnL vs Position buckets:")
            print(merged.groupby("pos_bucket", observed=False)["pnl_change"].mean())

        # -------------------------
        # Trade-level diagnostics
        # -------------------------
        if trade_state_df is not None and not trade_state_df.empty:
            print("\nTrade Edge:")
            print(f"Mean true edge: {trade_state_df['true_edge'].mean():.4f}")
            print(f"Std true edge: {trade_state_df['true_edge'].std():.4f}")

            print("\nMarkout after fills:")
            for h in [1, 3, 5, 10]:
                print(f"Mean markout_{h}: {trade_state_df[f'markout_{h}'].mean():.4f}")

            # Existing bucket diagnostics
            trade_state_df["trade_imb_bucket"] = pd.qcut(
                trade_state_df["imbalance_1"], 5, labels=False, duplicates="drop"
            )
            print("\nTrade edge vs top-of-book imbalance bucket:")
            print(trade_state_df.groupby("trade_imb_bucket")["true_edge"].mean())

            trade_state_df["trade_imb12_bucket"] = pd.qcut(
                trade_state_df["imbalance_12"], 5, labels=False, duplicates="drop"
            )
            print("\nTrade edge vs 2-level imbalance bucket:")
            print(trade_state_df.groupby("trade_imb12_bucket")["true_edge"].mean())

            trade_state_df["trade_imb3_bucket"] = pd.qcut(
                trade_state_df["imbalance_3"], 5, labels=False, duplicates="drop"
            )
            print("\nTrade edge vs 3-level imbalance bucket:")
            print(trade_state_df.groupby("trade_imb3_bucket")["true_edge"].mean())

            trade_state_df["trade_micro_bucket"] = pd.qcut(
                trade_state_df["microprice_offset"], 5, labels=False, duplicates="drop"
            )
            print("\nTrade edge vs microprice offset bucket:")
            print(trade_state_df.groupby("trade_micro_bucket")["true_edge"].mean())

            trade_state_df["trade_front_bid_bucket"] = pd.qcut(
                trade_state_df["front_share_bid"], 5, labels=False, duplicates="drop"
            )
            trade_state_df["trade_front_ask_bucket"] = pd.qcut(
                trade_state_df["front_share_ask"], 5, labels=False, duplicates="drop"
            )

            print("\nTrade edge vs bid front-share bucket:")
            print(trade_state_df.groupby("trade_front_bid_bucket")["true_edge"].mean())

            print("\nTrade edge vs ask front-share bucket:")
            print(trade_state_df.groupby("trade_front_ask_bucket")["true_edge"].mean())

            trade_state_df["trade_depth_slope_bid_bucket"] = pd.qcut(
                trade_state_df["depth_slope_bid"], 5, labels=False, duplicates="drop"
            )
            trade_state_df["trade_depth_slope_ask_bucket"] = pd.qcut(
                trade_state_df["depth_slope_ask"], 5, labels=False, duplicates="drop"
            )

            print("\nTrade edge vs bid depth slope bucket:")
            print(trade_state_df.groupby("trade_depth_slope_bid_bucket")["true_edge"].mean())

            print("\nTrade edge vs ask depth slope bucket:")
            print(trade_state_df.groupby("trade_depth_slope_ask_bucket")["true_edge"].mean())

            for h in [1, 3, 5]:
                trade_state_df[f"future_bucket_{h}"] = pd.qcut(
                    trade_state_df[f"future_move_{h}"], 5, labels=False, duplicates="drop"
                )
                print(f"\nTrue edge vs future move bucket ({h}) (down → up):")
                print(trade_state_df.groupby(f"future_bucket_{h}")["true_edge"].mean())

            trade_state_df["price_bucket"] = pd.qcut(
                trade_state_df["price"], 5, labels=False, duplicates="drop"
            )
            print("\nTrade markout_5 vs trade price bucket:")
            print(trade_state_df.groupby("price_bucket")["markout_5"].mean())

            print("\nMarkout_5 vs 3-level imbalance bucket:")
            print(trade_state_df.groupby("trade_imb3_bucket")["markout_5"].mean())

            print("\nMarkout_5 vs microprice offset bucket:")
            print(trade_state_df.groupby("trade_micro_bucket")["markout_5"].mean())

            print("\nMarkout_5 vs bid front-share bucket:")
            print(trade_state_df.groupby("trade_front_bid_bucket")["markout_5"].mean())

            print("\nMarkout_5 vs ask front-share bucket:")
            print(trade_state_df.groupby("trade_front_ask_bucket")["markout_5"].mean())

            print("\nMarkout_5 vs bid depth slope bucket:")
            print(trade_state_df.groupby("trade_depth_slope_bid_bucket")["markout_5"].mean())

            print("\nMarkout_5 vs ask depth slope bucket:")
            print(trade_state_df.groupby("trade_depth_slope_ask_bucket")["markout_5"].mean())

            # -------------------------
            # Fill selection quality vs local regime expectation
            # Coarser benchmark to avoid overfitting
            # -------------------------
            fill_df = trade_state_df.copy()
            fill_df["side"] = fill_df["signed_qty"].apply(lambda x: "BUY" if x > 0 else "SELL")

            fill_df["reg_vol_3"] = pd.qcut(fill_df["rolling_vol_50"], 3, labels=False, duplicates="drop")
            fill_df["reg_imb3_3"] = pd.qcut(fill_df["imbalance_3"], 3, labels=False, duplicates="drop")
            fill_df["reg_micro_3"] = pd.qcut(fill_df["microprice_offset"], 3, labels=False, duplicates="drop")

            regime_cols = ["side", "reg_vol_3", "reg_imb3_3", "reg_micro_3"]

            regime_stats = (
                fill_df.groupby(regime_cols, observed=False)
                .agg(
                    regime_count=("true_edge", "count"),
                    regime_edge_sum=("true_edge", "sum"),
                    regime_markout5_sum=("markout_5", "sum"),
                )
                .reset_index()
            )

            fill_df = fill_df.merge(regime_stats, on=regime_cols, how="left")

            fill_df["expected_true_edge_loo"] = (
                    (fill_df["regime_edge_sum"] - fill_df["true_edge"]) /
                    (fill_df["regime_count"] - 1)
            )
            fill_df["expected_markout_5_loo"] = (
                    (fill_df["regime_markout5_sum"] - fill_df["markout_5"]) /
                    (fill_df["regime_count"] - 1)
            )

            valid_mask = fill_df["regime_count"] >= 4
            valid_fill_df = fill_df[valid_mask].copy()

            if not valid_fill_df.empty:
                valid_fill_df["edge_shortfall"] = (
                        valid_fill_df["true_edge"] - valid_fill_df["expected_true_edge_loo"]
                )
                valid_fill_df["markout5_shortfall"] = (
                        valid_fill_df["markout_5"] - valid_fill_df["expected_markout_5_loo"]
                )

                print("\nFill selection quality vs local regime expectation (leave-one-out):")
                print(f"Comparable fills used: {len(valid_fill_df)} / {len(fill_df)}")
                print(f"Mean edge shortfall: {valid_fill_df['edge_shortfall'].mean():.4f}")
                print(f"Mean markout_5 shortfall: {valid_fill_df['markout5_shortfall'].mean():.4f}")
                print(f"Pct fills below expected edge: {(valid_fill_df['edge_shortfall'] < 0).mean():.4f}")
                print(f"Pct fills below expected markout_5: {(valid_fill_df['markout5_shortfall'] < 0).mean():.4f}")

                print("\nWorst fill regimes by expected shortfall:")
                worst_regimes = (
                    valid_fill_df.groupby(regime_cols, observed=False)[["edge_shortfall", "markout5_shortfall"]]
                    .mean()
                    .sort_values(["edge_shortfall", "markout5_shortfall"])
                    .head(8)
                )
                print(worst_regimes)

                cols_to_show = [
                    "timestamp", "side", "price", "mid_price_calc", "spread",
                    "rolling_vol_50", "imbalance_1", "imbalance_3", "microprice_offset",
                    "true_edge", "expected_true_edge_loo", "edge_shortfall",
                    "markout_5", "expected_markout_5_loo", "markout5_shortfall", "regime_count"
                ]

                print("\nWorst individual fills by edge shortfall:")
                print(
                    valid_fill_df[cols_to_show]
                    .sort_values("edge_shortfall")
                    .head(10)
                    .to_string(index=False)
                )

                print("\nBest individual fills by edge shortfall:")
                print(
                    valid_fill_df[cols_to_show]
                    .sort_values("edge_shortfall", ascending=False)
                    .head(10)
                    .to_string(index=False)
                )
            else:
                print("\nFill selection quality vs local regime expectation:")
                print("Not enough comparable fills after leave-one-out filtering.")

            # -------------------------
            # Buy / sell side split
            # -------------------------
            print("\nSide split:")
            side_stats = (
                trade_state_df.assign(side=trade_state_df["signed_qty"].apply(lambda x: "BUY" if x > 0 else "SELL"))
                .groupby("side")[["true_edge", "markout_1", "markout_3", "markout_5", "markout_10"]]
                .mean()
            )
            print(side_stats)

        # -------------------------
        # Trade Frequency
        # -------------------------
        if self.trades_df is not None:
            trades = self.trades_df[self.trades_df["symbol"] == product]
            trade_count = len(trades)
            print(f"\nTotal trades: {trade_count}")
            print(f"Trades per 1000 steps: {trade_count / len(df) * 1000:.2f}")

        # -------------------------
        # Inventory Bias
        # -------------------------
        if pos_df is not None and not pos_df.empty:
            avg_pos = pos_df["position"].mean()
            max_pos = pos_df["position"].max()
            min_pos = pos_df["position"].min()
            avg_abs_pos = pos_df["position"].abs().mean()

            print("\nInventory stats:")
            print(f"Avg position: {avg_pos:.2f}")
            print(f"Avg abs position: {avg_abs_pos:.2f}")
            print(f"Max position: {max_pos}")
            print(f"Min position: {min_pos}")

        # -------------------------
        # Correlations
        # -------------------------
        corr_vol = df["pnl_change"].corr(df["rolling_vol_50"])
        corr_spread = df["pnl_change"].corr(df["spread"])
        corr_imb = df["pnl_change"].corr(df["imbalance_1"])
        corr_imb12 = df["pnl_change"].corr(df["imbalance_12"])
        corr_imb3 = df["pnl_change"].corr(df["imbalance_3"])
        corr_micro = df["pnl_change"].corr(df["microprice_offset"])
        corr_front_bid = df["pnl_change"].corr(df["front_share_bid"])
        corr_front_ask = df["pnl_change"].corr(df["front_share_ask"])

        print("\nCorrelations:")
        print(f"PnL vs Vol: {corr_vol:.4f}")
        print(f"PnL vs Spread: {corr_spread:.4f}")
        print(f"PnL vs Imbalance_1: {corr_imb:.4f}")
        print(f"PnL vs Imbalance_12: {corr_imb12:.4f}")
        print(f"PnL vs Imbalance_3: {corr_imb3:.4f}")
        print(f"PnL vs Microprice offset: {corr_micro:.4f}")
        print(f"PnL vs Bid front-share: {corr_front_bid:.4f}")
        print(f"PnL vs Ask front-share: {corr_front_ask:.4f}")

        # -------------------------
        # Classical performance stats
        # -------------------------
        returns = df["pnl_change"].dropna()
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = mean_ret / (std_ret + 1e-9)

        cum_pnl = df["profit_and_loss"].copy()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_drawdown = drawdown.min()

        win_rate = (returns > 0).mean()

        print("\nReturn stats:")
        print(f"Mean return: {mean_ret:.4f}")
        print(f"Std return: {std_ret:.4f}")
        print(f"Sharpe-like: {sharpe:.4f}")
        print(f"Win rate: {win_rate:.4f}")
        print(f"Max drawdown: {max_drawdown:.2f}")

        if self.trades_df is not None:
            trades = self.trades_df[self.trades_df["symbol"] == product]
            trade_count = len(trades)
            pnl_per_trade = total_pnl / (trade_count + 1e-9)
            print(f"PnL per trade: {pnl_per_trade:.4f}")

        if pos_df is not None and not pos_df.empty:
            inv_efficiency = total_pnl / (avg_abs_pos + 1e-9)
            print(f"PnL per unit avg abs inventory: {inv_efficiency:.4f}")

        print("\n========================================\n")


# -------------------------
# USAGE
# -------------------------
if __name__ == "__main__":
    bt = Backtester("/Users/janekczajnik/Downloads/1780/1780.log")

    bt.load_log()
    bt.parse_activities()
    bt.parse_trades()
    bt.preprocess()

    bt.plot_product_dashboard("TOMATOES")
    bt.plot_product_dashboard("EMERALDS")
    bt.run_diagnostics("TOMATOES")
    bt.run_diagnostics("EMERALDS")