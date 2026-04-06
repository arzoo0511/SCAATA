
# SCAATA: Multi-Asset Algorithmic Trading Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Stable-Baselines3](https://img.shields.io/badge/Stable_Baselines3-000000?style=for-the-badge&logo=openai&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-API-green?style=for-the-badge)

## Overview
**SCAATA** is an advanced, end-to-end algorithmic trading system that leverages **Large Language Models (LLMs), Imitation Learning, and Reinforcement Learning**. 

Unlike traditional rule-based bots, SCAATA utilizes LLMs to dynamically mine algorithmic strategies from code repositories. It then pre-trains a hybrid neural network using Imitation Learning to mimic these expert strategies, before fine-tuning a **Recurrent Proximal Policy Optimization (RecurrentPPO)** agent to adapt to dynamic market environments.

## Key Features
- **Multi-Asset Portfolio Evaluation:** Backtests and evaluates against 6 major tech equities (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `META`).
- **LLM-Driven Strategy Extraction:** Uses the **Groq API** to scrape, normalize, and evaluate programmatic trading algorithms, building a unique repository of 79 baseline strategies.
- **Imitation Learning Warm-Start:** Prevents the RL agents "cold-start" problem by pre-training a meta-model on a 245,000+ sample expert-trajectory dataset (reducing initial policy loss by over 46%).
- **Deep Reinforcement Learning:** Deploys a RecurrentPPO agent across 100,000 market timesteps to capture temporal market dependencies.
- **Advanced Feature Engineering:** Calculates and standardizes robust market indicators including MACD, RSI, Rolling Volatility, Momentum, and Volume MAs.

---

## System Architecture

The pipeline is split into four distinct sequential phases:

1. **Data Engineering:** Historical market data ingestion followed by the calculation and normalization of non-stationary financial indicators.
2. **Strategy Mining:** Evaluates external rule-based strategies (mined via LLM) on the historical dataset to generate positive alpha "expert actions."
3. **Imitation Meta-Model:** Trains a supervised neural network to replicate the expert actions, achieving a high-fidelity baseline trading policy.
4. **RecurrentPPO Fine-Tuning:** The agent explores the action space using an RL environment wrapper, aiming to maximize risk-adjusted returns and manage drawdowns. 

---

## Installation & Setup

### Prerequisites
Ensure you have Python 3.10+ installed. It is recommended to use a virtual environment or Conda.

### Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install stable-baselines3 sb3-contrib gymnasium
pip install groq pandas numpy matplotlib yfinance
```

### Environment Variables
You will need a Groq API Key to run the LLM Strategy Extraction pipeline.
```bash
export GROQ_API_KEY="your_api_key_here"
```

---

## Usage

To run the pipeline, execute the main Jupyter Notebook which sequentially processes the steps:

```bash
jupyter notebook scaata_edit2.ipynb
```

**Pipeline Execution Steps:**
1. Initializing and downloading the TICKER data.
2. Running the `Groq` strategy web-scraper.
3. Constructing the `expert_dataset` (Outputs as `strategy_dataset.json`).
4. Training the `Imitation Model` (Saved to `imitation_model.pth`).
5. Running the `RecurrentPPO` RL Agent training over 100k timesteps.
6. Displaying final evaluation metrics.

---

## Evaluation & Benchmarks
During backtesting on the localized dataset (11,000+ timesteps), the system establishes baseline metrics utilized to map future hyperparameter variations. 

- **Timesteps Simulated:** 100,000
- **Total Action Space:** Discrete (Hold, Buy, Sell)
- **Features Processed:** 13 technical inputs
- **Imitation Loss Improvement:** ~46% reduction from initial epoch.

*(Note: Target returns and financial benchmarks are currently under active evaluation and optimization. Current iteration focuses on architectural routing vs. pure performance).*

---

## Disclaimer
**For Educational and Research Purposes Only.** 
SCAATA is an experimental algorithm. It does not constitute financial advice. The maintainers are not responsible for any financial losses incurred from deploying this code in live trading environments.
