"""
Downloads real daily data and saves two files:
  prices.csv  - 9 NSE stocks + USDINR, one row per trading day (used by var_backtest.py)
  usdinr.csv  - USD/INR only (used by exposure_simulator.py)

Stocks:  TejHQ Indian-markets dataset on Hugging Face (NSE end-of-day data from the official bhavcopy).
         We use adj_close, the price adjusted for splits, bonuses and dividends. The plain close price
         would show a split as a fake 50% crash and ruin the VaR numbers.
USD/INR: Yahoo Finance, or FRED if Yahoo does not work.

Run on a computer or Google Colab that has internet:
    pip install polars pandas yfinance pyarrow
    python get_real_data.py
"""
import datetime
import pandas as pd
import numpy as np

SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "ITC", "LT", "BHARTIARTL"]
START = "2015-01-01"
BASE = "hf://datasets/tejhq/indian-markets/prices_adjusted"


def stocks_to_wide(df):
    """Long table (date, symbol, adj_close) -> one column per stock."""
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last").sort_index()
    return wide[[s for s in SYMBOLS if s in wide.columns]]


def get_stocks():
    import polars as pl
    start = datetime.date.fromisoformat(START)
    keep = pl.col("symbol").is_in(SYMBOLS) & (pl.col("series") == "EQ") & (pl.col("date") >= start)
    try:                                              # read all NSE year files at once
        lf = pl.scan_parquet(f"{BASE}/nse_*.parquet")
        df = lf.filter(keep).select(["date", "symbol", "adj_close"]).collect().to_pandas()
    except Exception as e:                            # otherwise read year by year
        print("Reading all years at once failed (", e, "), trying year by year")
        parts = []
        for year in range(int(START[:4]), datetime.date.today().year + 1):
            try:
                part = (pl.scan_parquet(f"{BASE}/nse_{year}.parquet").filter(keep)
                        .select(["date", "symbol", "adj_close"]).collect().to_pandas())
                parts.append(part)
            except Exception as e2:
                print("Could not read", year, e2)
        df = pd.concat(parts)
    return stocks_to_wide(df)


def get_usdinr():
    try:
        import yfinance as yf
        fx = yf.download("USDINR=X", start=START, auto_adjust=True, progress=False)["Close"].squeeze()
        fx = fx.dropna()
        if len(fx) > 500:
            return fx.rename("USDINR")
    except Exception as e:
        print("Yahoo Finance failed:", e)
    print("Trying FRED (DEXINUS) instead")
    fx = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXINUS", index_col=0, parse_dates=True)
    fx = pd.to_numeric(fx.iloc[:, 0], errors="coerce").dropna()
    return fx[fx.index >= START].rename("USDINR")


def combine(stocks, fx):
    """Keep only days when the stock market was open; fill FX gaps with the previous value."""
    fx = fx.copy()
    fx.index = pd.to_datetime(fx.index).tz_localize(None)
    both = stocks.join(fx.reindex(stocks.index, method="ffill"))
    return both.dropna()


def main():
    stocks = get_stocks()
    print("Stocks found:", list(stocks.columns), "| missing:", [s for s in SYMBOLS if s not in stocks.columns])
    fx = get_usdinr()
    prices = combine(stocks, fx)
    print(prices.index.min().date(), "to", prices.index.max().date(), "|", len(prices), "days")

    # quick data check: a huge one-day move usually means a data problem (e.g. unadjusted split)
    big = np.log(prices / prices.shift(1)).abs().max()
    print("\nLargest one-day move per column (%):")
    print((100 * (np.exp(big) - 1)).round(1).to_string())
    print("Anything above about 20% deserves a look before you trust the results.")

    prices.to_csv("prices.csv")
    fx.to_frame().to_csv("usdinr.csv")
    print("\nSaved prices.csv and usdinr.csv")


if __name__ == "__main__":
    main()
