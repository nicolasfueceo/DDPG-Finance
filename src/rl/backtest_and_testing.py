import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PortfolioVisualizer:
    def __init__(self, portfolio_values, asset_prices, portfolio_weights_history=None):
        print("Initializing PortfolioVisualizer with:")
        print(f"Portfolio values shape: {np.array(portfolio_values).shape}")
        print(f"Asset prices shape/info: {type(asset_prices)}")
        if isinstance(asset_prices, pd.DataFrame):
            print(f"Asset prices columns: {asset_prices.columns}")
        print(
            f"Weights history shape: {None if portfolio_weights_history is None else np.array(portfolio_weights_history).shape}")

        self.portfolio_values = np.array(portfolio_values)
        self.asset_prices = asset_prices
        self.weights_history = portfolio_weights_history

    def plot_portfolio_performance(self, benchmark_strategies=None):
        fig, ax1 = plt.subplots(figsize=(15, 8))

        # Normalize portfolio values
        self.portfolio_values = self.portfolio_values / self.portfolio_values[0]

        # Plot DDPG portfolio performance
        ax1.plot(self.portfolio_values, label='DDPG Portfolio', linewidth=2, color='blue')

        # Plot benchmark strategies if provided
        if benchmark_strategies is not None:
            for label, values in benchmark_strategies.items():
                normalized_values = values / values[0]
                ax1.plot(normalized_values, label=label, linewidth=2)

            # Highlight best performing strategy
            best_strategy = max(benchmark_strategies,
                                key=lambda x: benchmark_strategies[x][-1] / benchmark_strategies[x][0])
            ax1.plot(benchmark_strategies[best_strategy] / benchmark_strategies[best_strategy][0],
                     label=f'Best: {best_strategy}', linewidth=2, color='orange')

        # Plot individual assets
        best_performance = -np.inf
        best_asset = None
        for column in self.asset_prices.columns:
            prices = self.asset_prices[column].values
            returns = prices / prices[0]
            ax1.plot(returns, alpha=0.2, linestyle='--')

            final_return = returns[-1]
            if final_return > best_performance:
                best_performance = final_return
                best_asset = column

        # Highlight best performing asset
        best_prices = self.asset_prices[best_asset].values
        best_returns = best_prices / best_prices[0]
        ax1.plot(best_returns, label=f'Best: {best_asset}', linewidth=2, color='orange')

        ax1.set_title('Portfolio Performance Comparison', fontsize=14, pad=20)
        ax1.set_xlabel('Time Steps', fontsize=12)
        ax1.set_ylabel('Cumulative Returns', fontsize=12)
        ax1.legend(fontsize=10, loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Add minor gridlines
        ax1.minorticks_on()
        ax1.grid(which='minor', linestyle=':', alpha=0.2)

        plt.tight_layout()
        return fig
    def plot_financial_metrics(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Rolling Sharpe Ratio
        window = 50
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        rolling_sharpe = pd.Series(returns).rolling(window, min_periods=1).apply(
            lambda x: np.sqrt(252) * x.mean() / (x.std() + 1e-8)  # Added epsilon to avoid div by zero
        )

        ax1.plot(rolling_sharpe, color='darkblue', linewidth=2)
        ax1.set_title('Rolling Sharpe Ratio (50-day)', fontsize=12)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Time Steps')

        # Drawdown Analysis
        running_max = np.maximum.accumulate(self.portfolio_values)
        drawdowns = (self.portfolio_values - running_max) / running_max
        ax2.fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.3)
        ax2.set_title('Portfolio Drawdown', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Drawdown %')

        plt.tight_layout()
        return fig

    def generate_performance_metrics(self):
        """Calculate and display key performance metrics"""
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]

        try:
            metrics = {
                'Total Return (%)': (self.portfolio_values[-1] / self.portfolio_values[0] - 1) * 100,
                'Annualized Sharpe Ratio': np.sqrt(252) * returns.mean() / (returns.std() + 1e-8),
                'Max Drawdown (%)': np.min((self.portfolio_values - np.maximum.accumulate(self.portfolio_values))
                                           / np.maximum.accumulate(self.portfolio_values)) * 100,
                'Annualized Volatility (%)': returns.std() * np.sqrt(252) * 100
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None

        return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

    def plot_portfolio_weights(self):
        if self.weights_history is None:
            print("No portfolio weights history available. Cannot plot portfolio weights.")
            return

        fig, ax = plt.subplots(figsize=(15, 8))
        num_assets = self.weights_history.shape[1]
        time_steps = range(len(self.weights_history))

        # Plot each asset's weight over time
        for i in range(num_assets):
            ax.plot(time_steps, self.weights_history[:, i], label=f"Asset {i}")

        ax.set_title("Portfolio Weights Over Time", fontsize=14, pad=20)
        ax.set_xlabel("Time Steps", fontsize=12)
        ax.set_ylabel("Portfolio Weight", fontsize=12)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)

        # Add minor gridlines
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', alpha=0.2)

        plt.tight_layout()
        return fig





if __name__ == "__main__":
    from src.data.preprocessing import load_multi_asset_data, add_btc_as_cash
    from src.rl.portfolio_gym import PortfolioEnv
    from src.rl.train_ddpg import backtest_ddpg
    from src.rl.ddpg_agent import DDPGAgent
    from src.data.preprocessing import create_price_tensor
    import os


    # Data loading and preprocessing
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)
    data_dir = os.path.join(parent_dir, "data/data")

    combined_data = load_multi_asset_data(data_dir)
    combined_data = add_btc_as_cash(combined_data)
    # normalized_data = normalize_prices(combined_data)

    # Split data
    backtest_periods = 50 * 48
    training_data = combined_data.iloc[:-backtest_periods]
    backtest_data = combined_data.iloc[-backtest_periods:]

    # Create tensors
    window_size = 50
    backtest_tensor = create_price_tensor(backtest_data, window_size)
    env_backtest = PortfolioEnv(backtest_tensor)

    # Initialize agent with same dimensions
    state_dim = (3, 50, 12)  # Your tensor shape (features, window, assets)
    action_dim = 12  # Number of assets
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=1000000,
        batch_size=50,
    )

    # Load pre-trained weights
    agent.load_weights("weights")

    # Run backtest
    portfolio_values_test, portfolio_weights_history = backtest_ddpg(agent, env_backtest)
    print("Backtesting completed. Final Portfolio Value:", portfolio_values_test[-1])

    # Add visualization
    visualizer = PortfolioVisualizer(
        portfolio_values=np.array(portfolio_values_test),
        asset_prices=combined_data.iloc[-len(portfolio_values_test):],
        portfolio_weights_history=np.array(portfolio_weights_history)
    )

    visualizer.plot_portfolio_performance()
    plt.savefig("results/portfolio_performance.png")
    plt.show()

    visualizer.plot_financial_metrics()
    plt.savefig("results/financial_metrics.png")
    plt.show()


    metrics = visualizer.generate_performance_metrics()
    print(metrics)


    visualizer.plot_portfolio_weights()
    plt.savefig("results/portfolio_weights.png")
    plt.show()





