import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from tabulate import tabulate
BINANCE_API_BASE = "https://api.binance.com/api/v3/klines"
BINANCE_API_24HR = "https://api.binance.com/api/v3/ticker/24hr"


def get_top_12_traded_pairs(base_currency="BTC"):
    """
    Fetch the top 12 traded cryptocurrency pairs by 24-hour trading volume where the base currency is specified (e.g., BTC).

    Parameters:
        base_currency (str): The base currency (default: "BTC").

    Returns:
        list: List of top 12 traded symbols (e.g., ["ETHBTC", "BNBBTC"]).
    """
    response = requests.get(BINANCE_API_24HR)

    if response.status_code != 200:
        raise Exception(f"Error fetching 24-hour stats: {response.json()}")

    data = response.json()
    df = pd.DataFrame(data)

    # Convert volume to float for sorting
    df["volume"] = df["volume"].astype(float)

    # Filter pairs with the specified base currency
    df = df[df["symbol"].str.endswith(base_currency)]
    # print coins sorted by volume
    print(tabulate(df.sort_values(by="volume", ascending=False), headers="keys", tablefmt="pretty"))
    # Sort by trading volume and get the top pairs
    top_pairs = df.sort_values(by="volume", ascending=False).head(11)["symbol"].tolist()


    # Ensure the base currency itself is included
    if f"{base_currency}{base_currency}" not in top_pairs:
        top_pairs.append(base_currency)

    return top_pairs

def fetch_binance_data(symbol, interval, start_time, end_time):
    """
    Fetch historical data from Binance API.

    Parameters:
        symbol (str): Cryptocurrency pair (e.g., "BTCUSDT").
        interval (str): Interval (e.g., "30m" for 30 minutes).
        start_time (int): Start time in milliseconds since epoch.
        end_time (int): End time in milliseconds since epoch.

    Returns:
        pd.DataFrame: DataFrame with high, low, close prices and timestamps.
    """
    url = f"{BINANCE_API_BASE}?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}&limit=1000"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Error fetching data for {symbol}: {response.json()}")

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

        try:
            data[symbol] = pd.DataFrame()
            start_time = end_time - timedelta(days=lookback_days)
            while start_time < end_time:
                start_time_str = int(start_time.timestamp() * 1000)
                end_time_str = int((start_time + timedelta(days=14)).timestamp() * 1000)
                print(f"Fetching {symbol} data from {start_time} to {end_time}")
                df = fetch_binance_data(symbol, interval, start_time_str, end_time_str)
                data[symbol] = pd.concat([data[symbol], df])
                start_time += timedelta(days=14)
        except Exception as e:
            print(f"Error fetching {symbol} data: {e}")
            continue


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


if __name__ == "__main__":
    # Fetch top 12 traded pairs
    top_pairs = get_top_12_traded_pairs("BTC")

    # Fetch historical data
    interval = "30m"
    lookback_days = 600

    #add btc to the list

    historical_data = get_historical_data(top_pairs, interval=interval, lookback_days=lookback_days)

    # Save data
    save_data(historical_data)
