import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import asyncio

BINANCE_API_BASE = "https://api.binance.com/api/v3/klines"


def fetch_binance_data(symbol, interval, start_time, end_time):
    """
    Fetch historical data from Binance API.

    Parameters:
        symbol (str): Cryptocurrency pair (e.g., "BTCUSDT").
        interval (str): Interval (e.g., "30m" for 30 minutes).
        start_time (str): Start time in milliseconds since epoch.
        end_time (str): End time in milliseconds since epoch.

    Returns:
        pd.DataFrame: DataFrame with high, low, close prices and timestamps.
    """
    url = f"{BINANCE_API_BASE}?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}&limit=1000"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.json()}")

    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]
    )
    df = df[["timestamp", "high", "low", "close"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["high", "low", "close"]] = df[["high", "low", "close"]].astype(float)

    return df


def get_historical_data(symbols, interval="30m", lookback_days=30):
    """
    Get historical data for multiple symbols.

    Parameters:
        symbols (list): List of cryptocurrency pairs (e.g., ["BTCUSDT", "ETHUSDT"]).
        interval (str): Interval for data (e.g., "30m").
        lookback_days (int): Number of days to look back.

    Returns:
        dict: Dictionary where keys are symbols and values are DataFrames.
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)

    # divide into 2 weeks at a time and asyncronously fetch data
    data = {}

    for symbol in symbols:
        data[symbol] = pd.DataFrame()
        start_time = end_time - timedelta(days=lookback_days)
        while start_time < end_time:
            start_time_str = int(start_time.timestamp() * 1000)
            end_time_str = int((start_time + timedelta(days=14)).timestamp() * 1000)
            print(f"Fetching {symbol} data from {start_time} to {end_time}")
            df = fetch_binance_data(symbol, interval, start_time_str, end_time_str)
            data[symbol] = pd.concat([data[symbol], df])
            start_time += timedelta(days=14)


    return data


def save_data(data, output_dir="data/"):
    """
    Save fetched data to CSV files.

    Parameters:
        data (dict): Dictionary of DataFrames.
        output_dir (str): Directory to save the CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for symbol, df in data.items():
        output_path = os.path.join(output_dir, f"{symbol}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {symbol} data to {output_path}")

