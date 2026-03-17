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

        df["mid_price_calc"] = (df["bid_price_1"] + df["ask_price_1"]) / 2
        df = df.sort_values(["product", "timestamp"]).reset_index(drop=True)
        self.activities_df = df

        if self.trades_df is not None and not self.trades_df.empty:
            trades = self.trades_df.copy()
            for col in ["timestamp", "price", "quantity"]:
                if col in trades.columns:
                    trades[col] = pd.to_numeric(trades[col], errors="coerce")
            trades = trades.sort_values("timestamp").reset_index(drop=True)
            self.trades_df = trades

    # -------------------------
    # FILTER PRODUCT
    # -------------------------
    def get_product_df(self, product: str):
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

        def signed_qty(row):
            buyer = row.get("buyer", "")
            seller = row.get("seller", "")

            if buyer == "SUBMISSION":
                return row["quantity"]
            if seller == "SUBMISSION":
                return -row["quantity"]
            return 0

        df["signed_qty"] = df.apply(signed_qty, axis=1)
        df["position"] = df["signed_qty"].cumsum()
        return df[["timestamp", "price", "quantity", "signed_qty", "position"]]

    # -------------------------
    # TRADE PRICE DISTRIBUTION
    # y-axis = price, x-axis = frequency
    # -------------------------
    # -------------------------
    # TRADE PRICE DISTRIBUTION
    # y-axis = price, x-axis = total buy/sell quantity
    # -------------------------
    def plot_trade_distribution_horizontal(self, ax, product: str):
        if self.trades_df is None or self.trades_df.empty:
            ax.set_title(f"{product} Trade Price Distribution (no trades)")
            return

        trade_df = self.trades_df[self.trades_df["symbol"] == product].copy()
        if trade_df.empty:
            ax.set_title(f"{product} Trade Price Distribution (no trades)")
            return

        trade_df = trade_df.dropna(subset=["price", "quantity"]).copy()
        if trade_df.empty:
            ax.set_title(f"{product} Trade Price Distribution (no prices)")
            return

        trade_df["price"] = trade_df["price"].astype(int)
        trade_df["quantity"] = pd.to_numeric(trade_df["quantity"], errors="coerce").fillna(0)

        def signed_qty(row):
            buyer = row.get("buyer", "")
            seller = row.get("seller", "")

            if buyer == "SUBMISSION":
                return row["quantity"]      # buys positive
            if seller == "SUBMISSION":
                return -row["quantity"]     # sells negative
            return 0

        trade_df["signed_qty"] = trade_df.apply(signed_qty, axis=1)

        buys = trade_df[trade_df["signed_qty"] > 0].groupby("price")["signed_qty"].sum()
        sells = -trade_df[trade_df["signed_qty"] < 0].groupby("price")["signed_qty"].sum()

        all_prices = sorted(set(buys.index).union(set(sells.index)))
        if not all_prices:
            ax.set_title(f"{product} Trade Price Distribution (no submission trades)")
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
    # BUILD 8-PANEL DASHBOARD
    # -------------------------
    def plot_dashboard(self):
        products = ["TOMATOES", "EMERALDS"]

        # IMPORTANT: do not share x across rows because row 4 is not timestamp-based
        fig, axs = plt.subplots(4, 2, figsize=(18, 12))

        for col_idx, product in enumerate(products):
            df = self.get_product_df(product)
            pos_df = self.compute_position(product)

            if df.empty:
                for row_idx in range(4):
                    axs[row_idx, col_idx].set_title(f"{product} - no data")
                continue

            # 1. PRICES
            axs[0, col_idx].plot(df["timestamp"], df["bid_price_1"], label="Bid 1")
            axs[0, col_idx].plot(df["timestamp"], df["ask_price_1"], label="Ask 1")
            axs[0, col_idx].plot(df["timestamp"], df["mid_price_calc"], label="Mid")
            axs[0, col_idx].set_title(f"{product} Prices")
            axs[0, col_idx].set_xlabel("Timestamp")
            axs[0, col_idx].set_ylabel("Price")
            axs[0, col_idx].legend(loc="upper right")

            # 2. PNL
            axs[1, col_idx].plot(df["timestamp"], df["profit_and_loss"])
            axs[1, col_idx].set_title(f"{product} PnL")
            axs[1, col_idx].set_xlabel("Timestamp")
            axs[1, col_idx].set_ylabel("PnL")

            # 3. POSITION
            if pos_df is not None and not pos_df.empty:
                axs[2, col_idx].step(pos_df["timestamp"], pos_df["position"], where="post")
                axs[2, col_idx].axhline(80, linestyle="--")
                axs[2, col_idx].axhline(-80, linestyle="--")
                axs[2, col_idx].axhline(0, linestyle=":")
                axs[2, col_idx].set_title(f"{product} Position")
                axs[2, col_idx].set_xlabel("Timestamp")
                axs[2, col_idx].set_ylabel("Position")
            else:
                axs[2, col_idx].set_title(f"{product} Position (no trades)")

            # 4. TRADE PRICE DISTRIBUTION
            self.plot_trade_distribution_horizontal(axs[3, col_idx], product)

        plt.tight_layout()
        plt.show()


# -------------------------
# USAGE
# -------------------------
if __name__ == "__main__":
    bt = Backtester("PATH TO OUTPUT.log FILE HERE")
    bt.load_log()
    bt.parse_activities()
    bt.parse_trades()
    bt.preprocess()
    bt.plot_dashboard()
