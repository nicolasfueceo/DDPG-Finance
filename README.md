# Deep Reinforcement Learning for Portfolio Management

This project implements a **Deep Deterministic Policy Gradient (DDPG)** framework for solving the financial portfolio management problem. Inspired by the works of Lillicrap et al. [1] and Jiang et al. [2], it leverages reinforcement learning to learn optimal asset allocations in a continuous action space, applying techniques like target networks and experience replay for stability.

## Motivation
Traditional portfolio management relies heavily on heuristic strategies, such as buy-and-hold or rule-based rebalancing. These methods often fail to adapt dynamically to the complexities and volatility of financial markets. 

Reinforcement learning provides a novel approach, enabling an agent to learn optimal portfolio allocations by interacting with the market environment and maximizing cumulative returns. This implementation extends DDPG to learn asset weights for cryptocurrency portfolios, leveraging price tensors, target networks, and Ornstein-Uhlenbeck noise for exploration.

---

## Project Structure

```
root
│
├── data/                # Directory containing the processed market data
│   ├── data/            # Raw market data
│   ├── ...              # Processing methods using binance APO
│
├── src/                # Source code for the implementation
│   ├── rl/             # Reinforcement Learning components
│      ├── actor_critic.py     # Actor and Critic network architectures
│      ├── ddpg_agent.py       # DDPG agent logic (training and action selection)
│      ├── replay_buffer.py    # Experience replay buffer
│      ├── portfolio_gym.py    # Portfolio management environment
│      ├── backtest_and_testing.py # Backtesting logic and benchmark comparisons
│      ├── train_ddpg.py # main training loop for agen 
│  
│
├── requirements.txt    # Python dependencies

```

---

## Installation

### Required Packages
Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

**Key Libraries:**
- `torch`: Deep learning framework
- `numpy`: Numerical computations
- `gym`: Reinforcement learning environment framework
- `matplotlib`: Visualization of results
- `pandas`: Data preprocessing
- `tensorboard`: Monitoring training

---

## Usage

### Training
To train the DDPG agent on cryptocurrency data:

```bash
python src/rl/train_ddpg.py
```

### Backtesting and Benchmarks
Evaluate the trained agent against benchmark strategies (e.g., uniform allocation or buy-and-hold):

```bash
python src/rl/backtest_and_testing.py
```

---

## References

[1] Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., … & Wierstra, D. (2016). Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971.

[2] Jiang, Z., Xu, D., & Liang, J. (2017). Deep Portfolio Management: A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem. arXiv preprint arXiv:1706.10059.

[3] Ghali, B. F. (n.d.). ddpg-rl-portfolio-management. GitHub. https://github.com/bassemfg/ddpg-rl-portfolio-management

[4] Polanco, A. (n.d.). Deep-Reinforcement-Learning-for-Optimal-Execution-of-Portfolio-Transactions-using-DDPG. GitHub. https://github.com/apolanco3225/Deep-Reinforcement-Learning-for-Optimal-Execution-of-Portfolio-Transactions-using-DDPG
