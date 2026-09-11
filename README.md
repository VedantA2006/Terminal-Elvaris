# Terminal-Elvaris: Quantitative Strategy Terminal & Multi-Agent Research Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Zero Lookahead Bias](https://img.shields.io/badge/AST%20Guard-Zero%20Lookahead-brightgreen.svg)]()
[![Dukascopy Real Data](https://img.shields.io/badge/Dataset-Dukascopy%205m%20XAUUSD-orange.svg)]()

An institutional quantitative trading terminal and autonomous multi-agent research lab for XAUUSD (Gold). Designed with TradingView Lightweight Charts, real-world broker execution friction ($0.20 spread + $0.05 slippage), hardcoded AST anti-lookahead bias validation, 1,000-path Monte Carlo stress testing, and a 4-agent collaborative research tournament.

---

## 🌟 Key Architecture & Capabilities

### 1. Autonomous Multi-Agent Quant Lab
Collaborative 4-agent tournament that iteratively generates, audits, optimizes, and stress-tests quantitative alpha models:
- **IdeaGeneratorAgent**: Mines combinatorial alpha across 16 diverse institutional archetypes (SMC sweeps, floor pivots, intraday VWAP, volatility squeezes, order block mitigations, Donchian channels, and volume displacement).
- **RiskOfficerAgent**: Statically audits candidate code with hardcoded AST lookahead inspection, enforces London/NY killzones, healthy institutional stop loss cushions (1.2x to 2.2x ATR, min $2.50 distance), and anti-bleed transition triggers.
- **ParameterGridSweeper**: Vectorized 9-point parameter optimization over real Dukascopy M5 bars to discover the alpha peak. Fast-prunes low-trade (< 5) models in ~2 seconds.
- **CriticPostMortem**: Diagnoses loss concentrations by session, volatility, and candle size.
- **OptimizerAgent**: Empirically refines strategy parameters and filters based on failure diagnostics.

### 2. High-Fidelity Backtesting Engine
- **Strict Candle-Close Execution**: Trades enter strictly on candle closes (`df['close']`), preventing unrealistic mid-candle fills.
- **Full Broker Friction Modeling**: Real-world spread ($0.20/oz) and slippage ($0.05/oz) applied to every entry and exit.
- **Intra-Bar Adverse Excursion**: Tracks true tick-level drawdown within bars to eliminate hidden intra-bar drawdowns.
- **Rigorous Train / Validation / Test Splitting**: 70/15/15 chronological split with an automated Validation Gate (rejects candidates with validation profit factor < 1.0 or net return < 0).
- **Monte Carlo Sequence-Risk Bootstrap**: 1,000-path trade-order shuffling evaluating maximum drawdown distribution, Risk of Ruin, and Probability of Profit.

### 3. TradingView Visual Terminal
- Interactive Lightweight Charts with candle-by-candle trade markers, entry/exit arrows, and PnL annotations.
- Floating Monthly Results breakdown widget showing Net R, dollar PnL, win rate, and trade counts for every calendar month.
- Quantitative Strategy Leaderboard ranking strategies by risk-adjusted return, high-yield months ($\ge +10\text{R}$), out-of-sample validation metrics, and Monte Carlo sequence VaR.

---

## 🚀 Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/VedantA2006/Terminal-Elvaris.git
cd Terminal-Elvaris

# Install dependencies
pip install -r requirements.txt
```

### Configure Environment Variables

Copy the template and set your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
OMNIROUTE_API_KEY=your_omniroute_api_key_here
OMNIROUTE_BASE_URL=http://localhost:20128/v1
PORT=5000
DEBUG=False
```

### Run Locally

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000/`.

---

## 📁 Repository Structure

```
├── app.py                      # Flask web application & API endpoints
├── data_split.py               # 70/15/15 chronological Train/Val/Test data splitter
├── stats_utils.py              # Shared risk-adjusted stats (Sharpe/Sortino calculation)
├── security_guard.py           # AST security validator & safe builtins whitelist
├── strategy.py                 # Active strategy module
├── strategy_executor.py        # Dynamic strategy executor with AST & security sandboxing
├── backtest.py                 # Bar-by-bar backtest simulation engine
├── default_strategy.py         # Default baseline strategy (Elvaris River V2)
├── ai_generator.py             # LLM strategy generator with hardcoded system prompts
├── ai_research_agents.py       # Multi-agent quant research pipeline (16 Archetypes)
├── autonomous_research_loop.py # Background autonomous research tournament manager
├── leaderboard.py              # Leaderboard persistence, ranking, and scoring
├── monte_carlo.py              # Monte Carlo sequence-risk bootstrap resampling
├── lookahead_guard.py          # AST-based static lookahead bias validator
├── download_data.py            # Dukascopy historical data loader with synthetic fallbacks
├── data/
│   ├── XAUUSD_5min.csv         # 34,744 Dukascopy 5m Gold candles
│   └── leaderboard.json        # Persistent ranked strategy registry
├── templates/
│   └── index.html              # Modern dark-mode TradingView terminal UI
└── static/
    ├── app.js                  # Frontend terminal controller & chart rendering
    └── style.css               # Terminal design system & styling
```

---

## 🛡️ Zero Lookahead Bias & Security Policy

All strategies are statically validated prior to execution:
- **Causality Check (`lookahead_guard.py`)**: Prohibits `shift(-n)`, negative indexing, `center=True` in rolling windows, and `.bfill()`.
- **Security Sandbox (`security_guard.py`)**: Restricts Python builtins, blocks `os`, `sys`, `subprocess`, socket operations, and dangerous dunder attribute access.
- **Out-of-Sample Validation (`data_split.py`)**: Ensures model fitting is restricted to the training window, gated by independent validation data before leaderboard qualification.
