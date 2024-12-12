import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from src.rl.ddpg_agent import DDPGAgent
from src.rl.portfolio_gym import PortfolioEnv
from src.data.preprocessing import create_price_tensor, load_multi_asset_data, add_btc_as_cash, normalize_prices


def train_ddpg(agent, env, num_episodes=1, max_steps=40, log_interval=10):
    run_name = f"DDPG_Run_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    initial_value = env.initial_balance

    # Noise configuration
    initial_noise_scale = 0.1
    noise_scale = initial_noise_scale
    noise_decay = 0.995  # Decay noise every episode by this factor
    min_noise_scale = 0.01

    for episode in range(num_episodes):
        state = env.reset()
        state_tensor = torch.tensor(state["price_tensor"], dtype=torch.float32).unsqueeze(0)
        print(state_tensor.shape)
        pws = torch.tensor(state["portfolio_weights"], dtype=torch.float32).unsqueeze(0)
        agent.actor.train()

        episode_rewards = []
        for step in range(max_steps):
            action = agent.select_action(state_tensor, pws, noise_scale=noise_scale)
            next_state, reward, done, info = env.step(action)

            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

            # Update state
            state = next_state
            state_tensor = torch.tensor(next_state["price_tensor"], dtype=torch.float32).unsqueeze(0)
            pws = torch.tensor(next_state["portfolio_weights"], dtype=torch.float32).unsqueeze(0)

            episode_rewards.append(reward)

            # Log portfolio value on a step-basis if desired
            step_idx = episode * max_steps + step
            writer.add_scalar("Log Portfolio Value/Step", np.log(info["portfolio_value"] / initial_value), step_idx)

            if done:
                break

        # Compute average and total return for this episode
        avg_episode_return = np.mean(episode_rewards)
        total_episode_return = np.sum(episode_rewards)

        # Log average return per episode
        writer.add_scalar("Return/AvgEpisodeReturn", avg_episode_return, episode)
        writer.add_scalar("Return/TotalEpisodeReturn", total_episode_return, episode)
        writer.add_scalar("Portfolio Value/Episode", info["portfolio_value"], episode)

        # Decay noise after each episode
        noise_scale = max(noise_scale * noise_decay, min_noise_scale)

        # Save weights periodically
        if (episode + 1) % log_interval == 0:
            os.makedirs("models/2", exist_ok=True)
            torch.save(agent.actor.state_dict(), f"models/4/ddpg_actor_episode_{episode + 1}.pth")
            torch.save(agent.critic.state_dict(), f"models/4/ddpg_critic_episode_{episode + 1}.pth")

    writer.close()


def backtest_ddpg(agent, env, max_steps=500):
    state = env.reset()
    state_tensor = torch.tensor(state["price_tensor"], dtype=torch.float32).unsqueeze(0)
    pws = torch.tensor(state["portfolio_weights"], dtype=torch.float32).unsqueeze(0)
    portfolio_values = []
    portfolio_weights_history   = []

    for step in range(max_steps):
        with torch.no_grad():
            action = agent.actor(state_tensor, pws)
            action = F.softmax(action, dim=-1).cpu().numpy()

        portfolio_weights_history.append(action)

        next_state, _, done, info = env.step(action)
        state_tensor = torch.tensor(next_state["price_tensor"], dtype=torch.float32).unsqueeze(0)
        pws = torch.tensor(next_state["portfolio_weights"], dtype=torch.float32).unsqueeze(0)
        portfolio_values.append(info["portfolio_value"])
        if done:
            break

    return portfolio_values, portfolio_weights_history


if __name__ == "__main__":
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)
    data_dir = os.path.join(parent_dir, "data/data")

    combined_data = load_multi_asset_data(data_dir)
    combined_data = add_btc_as_cash(combined_data)
    #normalized_data = normalize_prices(combined_data)

    backtest_periods = 50 * 48
    training_data = combined_data.iloc[:-backtest_periods]
    backtest_data = combined_data.iloc[-backtest_periods:]

    window_size = 50
    training_tensor = create_price_tensor(training_data, window_size)
    backtest_tensor = create_price_tensor(backtest_data, window_size)

    env_train = PortfolioEnv(training_tensor)
    env_backtest = PortfolioEnv(backtest_tensor)

    state_dim = training_tensor.shape[1:]
    action_dim = training_tensor.shape[-1]

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

    train_ddpg(agent, env_train)
    # Save weights and store original state
    print("\nSaving weights and testing save/load consistency:")

    # check if weights directory exists
    os.makedirs("weights", exist_ok=True)
    agent.save_weights("weights")
    original_weights = agent.actor.state_dict()

    # Load weights
    agent.load_weights("weights")
    loaded_weights = agent.actor.state_dict()

    # Compare weights
    print("\nComparing network parameters:")
    for (k1, v1), (k2, v2) in zip(original_weights.items(), loaded_weights.items()):
        print(f"Layer {k1}: Equal = {torch.equal(v1, v2)}")

    # Continue with backtesting
    portfolio_values_test = backtest_ddpg(agent, env_backtest)
    print("\nBacktesting completed. Final Portfolio Value:", portfolio_values_test[-1])
