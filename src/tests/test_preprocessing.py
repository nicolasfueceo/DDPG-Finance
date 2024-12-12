import pandas as pd
import numpy as np
from src.data.preprocessing import load_multi_asset_data, normalize_prices, create_price_tensor


def test_load_multi_asset_data():
    # Dummy data for two assets
    btc_data = pd.DataFrame({
        "timestamp": pd.date_range(start="2022-01-01", periods=100, freq="H"),
        "high": np.random.rand(100) * 50000,
        "low": np.random.rand(100) * 49000,
        "close": np.random.rand(100) * 49500,
    })

    eth_data = pd.DataFrame({
        "timestamp": pd.date_range(start="2022-01-01", periods=100, freq="H"),
        "high": np.random.rand(100) * 3000,
        "low": np.random.rand(100) * 2900,
        "close": np.random.rand(100) * 2950,
    })

    btc_data.to_csv("src/tests/data/BTC.csv", index=False)
    eth_data.to_csv("src/tests/data/ETH.csv", index=False)

    combined_data = load_multi_asset_data("src/tests/data/")
    print(combined_data.head())
    assert combined_data.shape[1] == 6, "Expected 6 columns for High, Low, Close of 2 assets"
    assert "High_BTC" in combined_data.columns, "Missing High_BTC column"
    assert "Close_ETH" in combined_data.columns, "Missing Close_ETH column"

    print("Load Multi-Asset Data Test Passed!")


def test_normalize_prices():
    # Dummy combined data
    data = pd.DataFrame({
        "High_BTC": [1000, 1050],
        "Low_BTC": [900, 950],
        "Close_BTC": [950, 1000],
        "High_ETH": [200, 210],
        "Low_ETH": [180, 190],
        "Close_ETH": [190, 200],
    })

    normalized_data = normalize_prices(data)
    print(normalized_data)
    assert normalized_data["Close_BTC"].iloc[-1] == 1.0, "Close_BTC at latest step should normalize to 1.0"
    assert normalized_data["Close_ETH"].iloc[-1] == 1.0, "Close_ETH at latest step should normalize to 1.0"

    print("Normalize Prices Test Passed!")


def test_create_price_tensor():
    # Dummy normalized data
    from tabulate import tabulate
    data = pd.DataFrame({
        "High_BTC": np.random.rand(100),
        "Low_BTC": np.random.rand(100),
        "Close_BTC": np.random.rand(100),
        "High_ETH": np.random.rand(100),
        "Low_ETH": np.random.rand(100),
        "Close_ETH": np.random.rand(100),
    })

    window_size = 10
    price_tensor = create_price_tensor(data, window_size)
    print(price_tensor.shape)
    print(tabulate(price_tensor[0, :, :, 0], headers=data.columns.str.split("_").str[0], showindex="always"))
    assert price_tensor.shape == (90, 3, 10, 2), "Expected shape (90, 3, 10, 2)"
    assert np.all(price_tensor[:, 0, :, :] <= 1), "Tensor should be normalized"

    print("Create Price Tensor Test Passed!")


if __name__ == "__main__":
    test_load_multi_asset_data()
    test_normalize_prices()
    test_create_price_tensor()
