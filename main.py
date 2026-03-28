import yfinance as yf
import pandas as pd

# Download daily data
df = yf.download("INFY.NS", period="1y", auto_adjust=False, progress=False)

# yfinance uses the date as the index, so make it a column if needed
df = df.reset_index()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

# Build weekly candles ending on Friday
weekly = df.resample("W-FRI").agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Adj Close": "last",
    "Volume": "sum"
})

# Remove the current incomplete week
current_week_end = pd.Timestamp.today().to_period("W-FRI").end_time.normalize()
weekly = weekly[weekly.index < current_week_end]

# Drop any empty rows just in case
weekly = weekly.dropna(subset=["Open", "High", "Low", "Close"])

# Optional: make it easier to read
weekly = weekly.reset_index().rename(columns={"Date": "week_end"})

print(weekly.head())
