import os
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def load_multi_asset_data(data_dir):
    """
    Load and combine price data for multiple assets.

    Parameters:
        data_dir (str): Path to the directory containing CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame with High, Low, Close columns for all assets.
    """
    all_data = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            asset_name = file.split(".")[0]  # Extract asset name from filename
            df = pd.read_csv(os.path.join(data_dir, file))
            df.rename(columns={
                'high': f'High_{asset_name}',
                'low': f'Low_{asset_name}',
                'close': f'Close_{asset_name}'
            }, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])  # Ensure time is datetime
            df.set_index('timestamp', inplace=True)
            all_data.append(df)

    # Combine all assets on the same time index
    combined_data = pd.concat(all_data, axis=1).dropna()
    return combined_data

def add_btc_as_cash(data):
    """
    Add BTCBTC (1:1 ratio for Bitcoin as cash) to the data.

    Parameters:
        data (pd.DataFrame): DataFrame containing price data.

    Returns:
        pd.DataFrame: DataFrame with BTCBTC added as the first asset.
    """
    btc_btc = pd.Series(1, index=data.index, name="BTCBTC")  # Fixed value of 1
    # Insert BTCBTC columns at the beginning
    data.insert(0, "High_BTCBTC", btc_btc)
    data.insert(1, "Low_BTCBTC", btc_btc)
    data.insert(2, "Close_BTCBTC", btc_btc)

    return data


def normalize_prices(data: pd.DataFrame):
    """
    Normalize High, Low, Close prices by the latest closing prices.

    Parameters:
        data (pd.DataFrame): Combined DataFrame with High, Low, Close columns for multiple assets.

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    close_cols = [col for col in data.columns if col.startswith("Close_")]
    latest_close = data[close_cols].iloc[-1]

    normalized_data = data.copy()
    for col in data.columns:
        asset_name = col.split("_")[1]
        normalized_data[col] = data[col] / latest_close[f"Close_{asset_name}"]

    return normalized_data




def create_price_tensor(data: pd.DataFrame, window_size: int = 50):
    """
    Create a price tensor for multiple assets, following the normalization scheme from the paper.
    Each window is normalized by its final closing prices.

    Parameters:
        data (pd.DataFrame): DataFrame with columns like 'High_Asset', 'Low_Asset', 'Close_Asset'.
                             All assets should have synchronized timestamps. The index is time.
        window_size (int): Number of periods in the history window.

    Returns:
        np.ndarray: A 4D tensor of shape (num_samples, 3, window_size, num_assets),
                    where the order of features is [Close, High, Low].
    """
    # Identify the assets by extracting the unique asset names from column headers
    # Columns are in form 'Close_AssetName', 'High_AssetName', etc.
    all_close_cols = [c for c in data.columns if c.startswith("Close_")]
    asset_names = [c.split("_", 1)[1] for c in all_close_cols]
    num_assets = len(asset_names)

    # Prepare lists to store tensor samples
    tensor_samples = []

    # We'll iterate from `window_size-1` to the end of the data so that we can form complete windows
    for i in range(window_size - 1, len(data)):
        window = data.iloc[i - window_size + 1: i + 1]

        # Ensure we have a full window
        if len(window) != window_size:
            continue

        # Extract price matrices
        close_matrix = window.filter(like="Close_").values.T  # shape: (num_assets, window_size)
        high_matrix = window.filter(like="High_").values.T    # shape: (num_assets, window_size)
        low_matrix = window.filter(like="Low_").values.T      # shape: (num_assets, window_size)

        # Normalization per paper: divide each series by the close price at the final timestep of that window
        final_closes = close_matrix[:, -1].reshape(-1, 1)  # shape: (num_assets, 1)
        # Avoid division by zero (shouldn't happen if data is valid, but just in case)
        final_closes[final_closes == 0] = 1e-8

        close_normalized = close_matrix / final_closes
        high_normalized = high_matrix / final_closes
        low_normalized = low_matrix / final_closes

        # Stack features in order [Close, High, Low]
        # Result shape: (3, num_assets, window_size)
        sample = np.stack([close_normalized, high_normalized, low_normalized], axis=0)

        # Now we have shape (3, num_assets, window_size)
        # The desired shape per the original code: (num_samples, 3, window_size, num_assets)
        # The current `sample` is (3, num_assets, window_size), we might want to swap axes
        # to match the desired final shape.
        # The paper's shape: (Samples, Features, Rolling Window, Assets)
        # Current shape: (3, num_assets, window_size)
        # We need to transpose to (3, window_size, num_assets)
        sample = sample.transpose(0, 2, 1)  # Now shape: (3, window_size, num_assets)

        tensor_samples.append(sample)

    # Convert list to a single numpy array
    # shape: (num_samples, 3, window_size, num_assets)
    if not tensor_samples:
        raise ValueError("No valid windows found. Check your data or window size.")

    price_tensor = np.array(tensor_samples)
    return price_tensor
