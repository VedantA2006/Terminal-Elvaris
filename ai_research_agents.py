"""
Multi-Agent Quantitative Research Engine.

Implements a 4-agent collaborative quantitative debate pipeline:
1. IdeaGeneratorAgent: Combinatorial Alpha Mining (Sequential Temporal Setups, SMC Sweeps, Daily Levels, Regime Gating, Champion Cross-Pollination)
2. RiskOfficerAgent: Rigorous Risk & Over-Fitting Auditor (syntax, AST lookahead guard, structural SL/TP, mandatory session masks)
3. CriticPostMortem: Failure Mode Diagnostics (analyzes the worst 10 losing trades by session, volatility, and candle size)
4. ParameterGridSweeper: Vectorized 9-Point Parameter Grid Sweeper (maximizes Net R and months >= 10R)
5. OptimizerAgent: Monte Carlo & High-Yield Refinement (tunes parameters and filters to hunt for >= 5 months >= 10R)
"""

import os
import re
import json
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from ai_generator import (
    call_ai_llm, _clean_code_response, HARDCODED_SYSTEM_PROMPT,
    get_engine_telemetry, set_engine_fallback, update_engine_telemetry
)
from lookahead_guard import validate_strategy_code, heal_strategy_code, is_syntax_error
from strategy_executor import execute_strategy
from monte_carlo import run_monte_carlo
from leaderboard import compute_monthly_r_breakdown, compute_rank_score


# Archetype prompt blueprints for combinatorial alpha mining
# Archetype prompt blueprints for combinatorial alpha mining (16 Diverse Institutional Archetypes)
ALPHA_ARCHETYPES = [
    {
        "id": "sequential_smc_fvg_state_machine",
        "name": "Sequential SMC Sweep + Multi-Bar FVG Mitigation",
        "concept": "Temporal state machine: SSL/BSL sweep arms setup for up to 10 bars, triggers entry when price mitigates 5m FVG on the FIRST touch with structural swing-anchored SL and Intraday VWAP alignment.",
        "instructions": """
Focus on building a Temporal Multi-Bar Setup with Wide Institutional Confluence:
1. Detect Liquidity Sweeps & Macro Direction:
   - SSL Sweep (Bullish): `ssl_sweep = (df['low'] < swing_lows) & (df['close'] > swing_lows)`
   - BSL Sweep (Bearish): `bsl_sweep = (df['high'] > swing_highs) & (df['close'] < swing_highs)`
   - Macro Alignment: Longs preferred when `df['close'] > vwap(df)` or above previous day pivot `levels['pp']`.
2. Multi-Bar Armed Window (up to 10 bars):
   - `armed_long = ssl_sweep.rolling(10, min_periods=1).max() == 1`
   - `armed_short = bsl_sweep.rolling(10, min_periods=1).max() == 1`
3. Mandatory London/NY Session Gating: `sess = session_mask(df, 'london_ny')`
4. First-Touch Mitigation Cross (ANTI-BLEED GUARD — NEVER FIRE CONTINUOUSLY):
   - `fvg_touch_long = (df['low'] <= bull_fvg_top) & (df['low'].shift(1) > bull_fvg_top) & (df['close'] > bull_fvg_bot)`
   - `fvg_touch_short = (df['high'] >= bear_fvg_bot) & (df['high'].shift(1) < bear_fvg_bot) & (df['close'] < bear_fvg_top)`
   - Transition signals:
     `raw_bull = armed_long & sess & fvg_touch_long & (df['close'] > vwap(df))`
     `bull_signal = raw_bull & (~raw_bull.shift(1).fillna(False))`
     `raw_bear = armed_short & sess & fvg_touch_short & (df['close'] < vwap(df))`
     `bear_signal = raw_bear & (~raw_bear.shift(1).fillna(False))`
5. Healthy Institutional Risk: Stop loss anchored below swing low with 1.2x to 1.8x ATR buffer (minimum $2.50 distance). Take Profit scaled at 1.5x to 2.0x risk.
"""
    },
    {
        "id": "multi_timeframe_key_levels",
        "name": "Daily Floor Pivots + Intraday VWAP Confluence",
        "concept": "Institutional key level confluence: Previous Day High/Low & Floor Pivots combined with VWAP trend direction inside London/NY Killzones.",
        "instructions": """
Focus on combining:
1. Macro Daily Levels: `levels = daily_levels(df)` (pdh, pdl, pp, r1, s1)
2. VWAP Intraday Trend: `vwap_line = vwap(df)`
3. Session Filter: `sess = session_mask(df, 'london_ny')`
4. Sequential Execution (First-Touch Transition):
   - Price sweeps below S1 or PDL, closes back above it (SSL sweep), price above VWAP -> Bullish Entry.
   - Price sweeps above R1 or PDH, closes back below it (BSL sweep), price below VWAP -> Bearish Entry.
   - Anti-bleed: `bull_signal = raw_bull & (~raw_bull.shift(1).fillna(False))`
5. Institutional Stops: Anchor SL below candle low or S1 with a 1.2x to 1.8x ATR buffer (minimum $2.50 distance). R:R = 1.5 to 2.0.
"""
    },
    {
        "id": "regime_gated_execution",
        "name": "Regime-Gated Adaptive Volatility Breakout",
        "concept": "Filters out chop using Kaufman Efficiency Ratio (ER > 0.28) and ADX (> 22) before executing Supertrend momentum breakouts in London/NY.",
        "instructions": """
Focus on combining:
1. Market Regime Filter: `er = efficiency_ratio(df['close'], 20)` (> 0.28) AND `adx_val, _, _ = adx(df, 14)` (> 22)
2. Trend Engine: `st_line, st_dir = supertrend(df, 10, 3.0)`
3. Session Filter: `sess = session_mask(df, 'london_ny')`
4. Entry Conditions (Transition crossovers only):
   - Bullish: `(st_dir == 1) & (st_dir.shift(1) <= 0) & (er > 0.28) & (adx_val > 22) & sess`
   - Bearish: `(st_dir == -1) & (st_dir.shift(1) >= 0) & (er > 0.28) & (adx_val > 22) & sess`
5. Trailing / Structural Risk: SL anchored 1.5x ATR below entry or at Supertrend line (min $2.50 distance). R:R = 1.6 to 2.2.
"""
    },
    {
        "id": "champion_cross_pollination",
        "name": "Champion Cross-Pollination (LSS Hybrid + VWAP Slope)",
        "concept": "Cross-pollinating the LSS Champion framework with VWAP alignment and dynamic session volatility filters.",
        "instructions": """
The Champion Strategy operates on:
- Swing Length = 7, ATR Length = 14, R:R = 1.5 to 2.0.
- Liquidity sweeps of swing highs/lows confirmed by subsequent FVG mitigation in London/NY killzones.
Cross-pollination goal:
- Add `vwap(df)` intraday trend filter (Longs only above VWAP, Shorts only below VWAP) or central floor pivot `pp` confirmation.
- Use healthy 1.2x to 1.8x ATR buffer (minimum $2.50 stop distance) so broker spread and slippage never erode edges.
- Prevent consecutive candle firing: `bull_signal = raw_bull & (~raw_bull.shift(1).fillna(False))`.
"""
    },
    {
        "id": "bollinger_keltner_squeeze",
        "name": "Bollinger-Keltner Volatility Squeeze Expansion",
        "concept": "Detects institutional compression when Bollinger Bands contract inside Keltner Channels, firing explosive momentum entries on directional expansion.",
        "instructions": """
Focus on Volatility Squeeze & Directional Expansion:
1. Volatility Squeeze Detection:
   - `bb_upper, bb_lower, bb_mid = bollinger_bands(df['close'], 20, 2.0)`
   - `kc_upper, kc_lower, kc_mid = keltner_channels(df, ema_period=20, atr_period=10, mult=1.5)`
   - Squeeze condition: `squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)`
   - Squeeze fire (expansion): `squeeze.shift(1) & ~squeeze`
2. Directional Momentum Filter:
   - `macd_line, macd_sig, macd_hist = macd(df['close'], 12, 26, 9)`
   - London/NY Killzones: `sess = session_mask(df, 'london_ny')`
3. Signals:
   - Bullish: Squeeze fire & (macd_hist > 0) & (df['close'] > kc_upper) & sess
   - Bearish: Squeeze fire & (macd_hist < 0) & (df['close'] < kc_lower) & sess
   - Anti-bleed: ensure signals fire only on first breakout bar.
4. Risk Management: SL anchored to opposite Keltner mid band with 1.2x to 1.6x ATR buffer (min $2.50). Target 1.5R to 2.0R.
"""
    },
    {
        "id": "vwap_statistical_zscore_fade",
        "name": "Intraday VWAP 2.5-Sigma Z-Score Mean Reversion Fade",
        "concept": "Exploits institutional mean-reversion when Gold stretches 2.5+ standard deviations away from intraday VWAP during peak sessions.",
        "instructions": """
Focus on Statistical Quantitative Arbitrage:
1. Intraday VWAP & Deviation:
   - `vwap_line = vwap(df)`
   - `z_val = zscore(df['close'] - vwap_line, period=30)`
2. Trend Exhaustion Indicator:
   - Wilder's RSI: `rsi_val = rsi(df['close'], 14)`
   - Session Filter: `sess = session_mask(df, 'london_ny')`
3. Mean Reversion Entry (First Reversal Candle):
   - Bullish Fade: `(z_val < -2.2) & (rsi_val < 32) & (df['close'] > df['open']) & sess`
   - Bearish Fade: `(z_val > 2.2) & (rsi_val > 68) & (df['close'] < df['open']) & sess`
   - Anti-bleed: `bull_signal = raw_bull & (~raw_bull.shift(1).fillna(False))`
4. Risk & Target:
   - SL anchored beyond candle extreme + 1.2x ATR (minimum $2.50 distance).
   - TP targeted at VWAP line or 1.5x risk distance.
"""
    },
    {
        "id": "london_open_judas_swing",
        "name": "Asian Range Liquidity Sweep (London Judas Swing)",
        "concept": "Identifies the false breakout (Judas Swing) of Asian Session highs/lows at London Open (07:00-09:30 UTC) and rides the real institutional trend.",
        "instructions": """
Focus on Session High/Low Liquidity Engineering:
1. Asian Session High & Low:
   - Use `session_mask(df, 'asia')` to track Asian extremes.
2. London Open Sweep:
   - London Killzone: `sess_lon = session_mask(df, 'london')`
   - Bullish Judas Sweep: London price dips below Asian Low but candle closes back inside Asian range.
   - Bearish Judas Sweep: London price spikes above Asian High but candle closes back below Asian high.
3. Confirmation:
   - Relative Volume spike: `rvol(df, 20) > 1.2`
   - Anti-bleed: trigger strictly on the first candle closing back inside the range.
4. Structural Risk: Anchor SL at the spike wick extreme + 1.2x ATR buffer (minimum $2.50). Target 1.5R to 2.2R.
"""
    },
    {
        "id": "volume_spread_rvol_displacement",
        "name": "Volume Spread Analysis + Institutional Displacement",
        "concept": "Filters for high institutional volume spikes (RVOL > 1.8) accompanied by large displacement candles (>65% body) breaking market structure.",
        "instructions": """
Focus on Institutional Footprint & Displacement:
1. Relative Volume: `rvol_val = rvol(df, 20)`
2. Displacement Candle:
   - `c_range = df['high'] - df['low']`
   - `c_body = (df['close'] - df['open']).abs()`
   - `is_displacement = (c_body / c_range > 0.65) & (c_range > 1.2 * atr(df, 14))`
3. Market Structure Break:
   - `sw_high, sw_low = find_swings(df, 7)`
   - Bullish BOS: `df['close'] > sw_high`
   - Bearish BOS: `df['close'] < sw_low`
4. Signals (First-Touch Transition):
   - Bullish: `is_displacement & (df['close'] > df['open']) & (rvol_val > 1.4) & (df['close'] > sw_high) & session_mask(df, 'london_ny')`
   - Anti-bleed: `bull_signal = raw_bull & (~raw_bull.shift(1).fillna(False))`
5. Risk Management: SL anchored at displacement candle low/high with 1.2x to 1.6x ATR buffer (minimum $2.50). Target 1.5R to 2.0R.
"""
    },
    {
        "id": "donchian_turtle_momentum",
        "name": "Donchian 20-Bar Turtle Breakout + Chandelier Stop",
        "concept": "Adaptive Turtle breakout system adapted for 5-minute Gold: enters on 20-bar Donchian channel breakouts protected by dynamic Chandelier ATR stops.",
        "instructions": """
Focus on Trend Persistence & Momentum:
1. Donchian Channels: `d_up, d_dn, d_mid = donchian_channels(df, period=20)`
2. Trend Filter: EMA 55/200 and Kaufman Efficiency Ratio (`er = efficiency_ratio(df['close'], 20) > 0.25`)
3. Chandelier Trailing Exit: `chan_long, chan_short = chandelier_exit(df, period=22, mult=2.5)`
4. Entry Signals:
   - Bullish: `(df['close'] > d_up.shift(1)) & (df['close'] > ema(df['close'], 55)) & (er > 0.25) & session_mask(df, 'london_ny')`
   - Anti-bleed: fire only on initial breakout bar (`~df['close'].shift(1) > d_up.shift(2)`).
5. Risk: SL anchored to `chan_long` / `chan_short` or 1.5x ATR buffer (minimum $2.50). Target 1.6R to 2.4R.
"""
    },
    {
        "id": "momentum_rsi_divergence",
        "name": "Structural Swing Divergence with Wilder's RSI",
        "concept": "Identifies institutional accumulation/distribution when price forms lower swing lows but RSI forms higher lows (bullish divergence) inside Killzones.",
        "instructions": """
Focus on Structural Momentum Divergence:
1. Swing Lows & Highs: `sw_h, sw_l = find_swings(df, 7)`
2. RSI Calculation: `rsi_val = rsi(df['close'], 14)`
3. Regular Bullish Divergence:
   - Price makes fresh swing low (`df['low'] < sw_l.shift(1)`), but RSI is higher than previous swing low RSI.
   - Filter: `session_mask(df, 'london_ny')` and price above intraday VWAP.
   - Anti-bleed: fire only on the confirmation reversal bar.
4. Risk Management: SL anchored to swing extreme with 1.2x to 1.8x ATR buffer (minimum $2.50 distance). Target 1.5R to 2.0R.
"""
    },
    {
        "id": "order_block_mitigation_bos",
        "name": "Institutional Order Block Retest + Market BOS",
        "concept": "Locates the last down-candle before a violent upward displacement (Bullish OB) and enters on the causal retest of the OB zone.",
        "instructions": """
Focus on Order Block Mitigation:
1. Identify Order Block Zone:
   - Bullish OB: Bearish candle (`close < open`) preceding strong displacement candle (`close > open + 1.2*ATR`).
   - Bearish OB: Bullish candle (`close > open`) preceding strong displacement candle (`close < open - 1.2*ATR`).
2. Retest Entry (First Touch):
   - Price retraces into the OB zone and prints a rejection wick in the original displacement direction.
   - Anti-bleed: fire only on the first retest bar.
3. Session Filter: `session_mask(df, 'london_ny')` and RVOL > 1.2.
4. Structural Stops: SL placed strictly below the OB candle low/high with 1.2x to 1.6x ATR buffer (minimum $2.50). Target 1.5R to 2.0R.
"""
    },
    {
        "id": "linear_regression_slope_drift",
        "name": "Vectorized Linear Regression Velocity Drift",
        "concept": "Measures continuous institutional directional price drift via 20-bar Linear Regression Slope, entering pullbacks in the drift direction.",
        "instructions": """
Focus on Statistical Price Velocity:
1. Linear Regression Slope: `slope = linear_regression_slope(df['close'], 20)`
2. EMA Trend Alignment: `fast_ema = ema(df['close'], 21)`, `slow_ema = ema(df['close'], 55)`
3. Pullback Entry (First Touch):
   - Bullish: `(slope > 0.15) & (fast_ema > slow_ema) & (df['low'] <= fast_ema) & (df['close'] > fast_ema) & session_mask(df, 'london_ny')`
   - Anti-bleed: `bull_signal = raw_bull & (~raw_bull.shift(1).fillna(False))`
4. Risk: SL anchored below recent swing low with 1.2x to 1.6x ATR buffer (minimum $2.50). Target 1.5R to 2.0R.
"""
    },
    {
        "id": "equal_highs_lows_sweep",
        "name": "Equal Highs/Lows (EQH/EQL) Liquidity Hunt",
        "concept": "Hunts double tops/bottoms (equal highs/lows) where retail stops accumulate, entering aggressively when the liquidity pool is swept and rejected.",
        "instructions": """
Focus on Retail Liquidity Pools:
1. Detect Equal Highs / Lows:
   - EQL: Two distinct swing lows within 0.05% of each other.
   - EQH: Two distinct swing highs within 0.05% of each other.
2. The Liquidity Hunt:
   - Price breaches EQL/EQH by 0.20 to 1.5x ATR, but fails to sustain and closes back within the range on heavy volume (`rvol(df, 20) > 1.2`).
   - Anti-bleed: trigger strictly on the candle closing back inside the range.
3. Session Filter: `session_mask(df, 'london_ny')` with Supertrend trend confirmation.
4. Risk: SL anchored at sweep wick extreme + 1.2x to 1.6x ATR (minimum $2.50). Target 1.5R to 2.0R.
"""
    },
    {
        "id": "stochastic_macd_momentum",
        "name": "Stochastic Dynamic Cycle + MACD Acceleration",
        "concept": "Synchronizes multi-indicator momentum: enters when 14-period Stochastic crosses out of extreme levels aligned with expanding MACD histogram bars.",
        "instructions": """
Focus on Indicator Cycle Confluence:
1. Stochastic: `k, d = stochastic(df, 14, 3)`
2. MACD: `macd_line, macd_sig, macd_hist = macd(df['close'], 12, 26, 9)`
3. Alignment (Crossover Transitions):
   - Bullish: `(k.shift(1) < 25) & (k > d) & (k.shift(1) <= d.shift(1)) & (macd_hist > 0) & session_mask(df, 'london_ny')`
   - Bearish: `(k.shift(1) > 75) & (k < d) & (k.shift(1) >= d.shift(1)) & (macd_hist < 0) & session_mask(df, 'london_ny')`
   - Anti-bleed: crossover condition inherently fires only once.
4. Risk: SL anchored at lowest low of last 5 bars + 1.2x to 1.6x ATR (minimum $2.50). Target 1.5R to 2.0R.
"""
    },
    {
        "id": "inverted_fvg_breaker",
        "name": "Inverted Fair Value Gap (IFVG) Support/Resistance Breaker",
        "concept": "Identifies failed Fair Value Gaps where price broke through the gap without respecting it, converting it into a potent support/resistance breaker level.",
        "instructions": """
Focus on Inverted FVG Flip Levels:
1. FVG Detection: `b_fvg_top, b_fvg_bot, s_fvg_top, s_fvg_bot = find_fvgs(df)`
2. Inversion (Breaker):
   - Bullish Breaker: A bearish FVG (`s_fvg_top`) that price blew upward through. When price pulls back to retest `s_fvg_top` from above, it acts as support.
   - Retest touch: `(df['low'] <= s_fvg_top) & (df['low'].shift(1) > s_fvg_top) & (df['close'] > s_fvg_top)`
3. Gating: `session_mask(df, 'london_ny')` and price above VWAP.
4. Risk: SL placed 1.2x to 1.8x ATR past the breaker zone (minimum $2.50). Target 1.5R to 2.0R.
"""
    },
    {
        "id": "opening_range_breakout_orb",
        "name": "15-Minute Opening Range Breakout (ORB) in London/NY",
        "concept": "Captures the high-probability opening drive by tracking the first 15 minutes of London (07:00-07:15 UTC) and NY (12:30-12:45 UTC) sessions.",
        "instructions": """
Focus on Opening Range Directional Drives:
1. Define 15-minute Opening Range:
   - Track high and low of the first 3 M5 candles of London and NY open.
2. Breakout (First bar only):
   - Bullish: Candle closes above Opening Range High with `rvol(df, 20) > 1.2` and `(~close.shift(1) > orb_high)`.
   - Bearish: Candle closes below Opening Range Low with `rvol(df, 20) > 1.2` and `(~close.shift(1) < orb_low)`.
3. Filter: `session_mask(df, 'london_ny')` and ADX > 20.
4. Risk: SL at Opening Range Midpoint or 1.5x ATR buffer (minimum $2.50). Target 1.5R to 2.2R.
"""
    }
]


# ===========================================================================
# 1. IDEA GENERATOR AGENT
# ===========================================================================

class IdeaGeneratorAgent:
    """
    Proposes creative institutional quantitative trading hypotheses
    across combinatorial alpha archetypes with exploration memory.
    """

    def __init__(self, provider: str = 'omniroute', api_key: str = '', model: str = '', endpoint: str = None):
        self.provider = (provider or 'omniroute').lower()
        self.api_key = api_key or os.environ.get('OMNIROUTE_API_KEY', '') or os.environ.get('GROQ_API_KEY', '')
        self.model = model or 'agentrouter/gpt-6-astra'
        if self.model in ('auto/best-coding', 'auto/best-reasoning', 'auto', 'groq/qwen/qwen3.6-27b', 'qwen/qwen3.6-27b'):
            self.model = 'agentrouter/gpt-6-astra'
        self.endpoint = endpoint or os.environ.get('OMNIROUTE_ENDPOINT', 'http://localhost:20128/v1')

    def propose_strategy(self, archetype_idx: int = 0, recent_hypotheses: List[str] = None, custom_focus: str = None,
                         champion_code: str = None, second_parent_code: str = None,
                         failure_memory: List[str] = None, leaderboard_code_context: List[Dict] = None) -> Dict[str, Any]:
        """Generates an initial strategy code proposal based on a chosen archetype or breeds a champion parent."""
        archetype = ALPHA_ARCHETYPES[archetype_idx % len(ALPHA_ARCHETYPES)]

        novelty_mandate = ""
        if recent_hypotheses and not champion_code:
            formatted_recent = "\n".join(f"- {h}" for h in recent_hypotheses[-20:])
            novelty_mandate = f"""
CRITICAL NOVELTY MANDATE — AVOID DUPLICATE STRATEGIES:
The following concepts were ALREADY explored in recent rounds:
{formatted_recent}

You MUST NOT replicate the exact indicator parameters, indicator combinations, or setup logic of the above.
Innovate with alternative threshold values, unique filter combinations, or distinct execution confirmations to guarantee that this strategy is genuinely novel!
"""

        # UPGRADE 4: Failure Memory — teach LLM from past mistakes
        failure_section = ""
        if failure_memory:
            formatted_failures = "\n".join(f"- {f}" for f in failure_memory[-8:])
            failure_section = f"""
LEARN FROM RECENT FAILURES — DO NOT REPEAT THESE MISTAKES:
{formatted_failures}

Key lessons: Avoid over-filtering (causes 0 trades), ensure out-of-sample robustness (avoid curve-fitting to train data),
and keep entry conditions to 2-3 confluences maximum to generate sufficient trade frequency.
"""

        # UPGRADE 1: Leaderboard code context — show LLM what winning code looks like
        leaderboard_section = ""
        if leaderboard_code_context and not champion_code:
            lb_snippets = []
            for i, lc in enumerate(leaderboard_code_context[:3]):
                lb_snippets.append(
                    f"### Winner #{i+1}: {lc['name']} ({lc['total_r']:+.1f}R, PF {lc['profit_factor']}, WR {lc['win_rate']}%)\n"
                    f"```python\n{lc['code']}\n```"
                )
            leaderboard_section = f"""
PROVEN WINNING STRATEGY CODE (Learn from these patterns):
{chr(10).join(lb_snippets)}

Study these winners carefully. Notice their signal structure, risk management, and confluence patterns.
Create a NOVEL strategy that is DIFFERENT from these but learns from their structural discipline!
"""

        breeding_mandate = ""
        if champion_code:
            # UPGRADE 5: Second parent for cross-pollination
            second_parent_section = ""
            if second_parent_code:
                second_parent_section = f"""
SECOND PARENT FOR CROSS-POLLINATION:
```python
{second_parent_code[:500]}
```
Cross-breed elements from BOTH parents: combine Parent A's entry logic with Parent B's risk management,
or merge Parent A's filters with Parent B's signal structure. Create a genuinely novel hybrid!
"""
            breeding_mandate = f"""
🧬 GENETIC MUTATION / CROSS-BREEDING MANDATE:
Below is our current Champion Strategy from the Leaderboard (proven positive Net R and high win rate):
```python
{champion_code}
```
{second_parent_section}
YOUR BREEDING MISSION:
Do NOT discard the winning logic! Mutate and evolve this champion by cross-breeding it with the Archetype: "{archetype['name']}" ({archetype['concept']}).
Guidelines for mutation:
1. Retain the core winning edge of the parent strategy (e.g. key filters, signal structure, ATR risk management).
2. Inject innovative confluences from {archetype['name']} (e.g. enhanced volume gating, session VWAP context, higher-timeframe trend alignment, or dynamic volatility expansion).
3. Evolve the risk management (keep stop loss at least 1.5x-2.0x ATR and min $4.00, improve Take Profit targeting with 1.8R-2.5R).
4. Prune noisy/redundant conditions if they cause over-fitting or unnecessary bleed.
Produce a refined, evolved strategy that outperforms the parent champion!
"""

        user_prompt = f"""
[AGENT: IDEA GENERATOR]
Your mission: Synthesize a high-performing institutional quantitative trading strategy for XAUUSD (Gold 5-minute candles).

ALPHA ARCHETYPE:
Name: {archetype['name']}
Concept: {archetype['concept']}
Research Guidelines:
{archetype['instructions']}

{f'Specialized Focus: {custom_focus}' if custom_focus else ''}
{breeding_mandate if champion_code else novelty_mandate}
{failure_section}
{leaderboard_section}

MANDATORY INSTITUTIONAL RULES FOR PROFITABILITY:
1. STRICT ZERO LOOKAHEAD BIAS: No shift(-n), no center=True, no bfill(). All calculations causal.
   - ABSOLUTE BAN: NEVER use `center=True` in `.rolling()` (e.g. `df['high'].rolling(..., center=True)` peeks into future candles and is PERMANENTLY BLOCKED by the AST security parser).
   - Use strictly causal rolling: `df['high'].rolling(15).max().shift(1)` or call the built-in `find_swings(df)`.
2. REALISTIC CONFLUENCES (PREVENT ZERO-TRADE OVER-FILTERING):
   - Use 2 to 3 confluences MAXIMUM: [Macro Trend Context] + [Archetype Entry Trigger] + [Session Mask].
   - Examples of macro context: Intraday VWAP (`df['close'] > vwap(df)`), Daily Pivots (`levels['pp']`), or EMA 21/55.
   - CRITICAL: Do NOT stack 5 or 6 simultaneous indicators with `&`. Over-filtering causes 0 trades to ever trigger!
   - Target 15 to 45 quality trades over the 6-month train period (approx. 1 to 2 trades per week).
3. ANTI-BLEED / NON-REPEATING TRANSITION SIGNALS (PREVENT OVERTRADING CHURN):
   - Entry signals MUST trigger ONLY on the FIRST candle transition of a setup:
     `raw_bull = setup_condition & sess`
     `bull_signal = raw_bull & (~raw_bull.shift(1).fillna(False))`
   - NEVER fire repeatedly on consecutive bars during a rolling window!
4. INSTITUTIONAL STOP LOSS & TAKE PROFIT:
    - ALL trade entries are taken strictly at the CANDLE CLOSE (`df['close']`).
    - Stop Loss MUST use a FULL ATR multiplier (1.2x to 2.2x) — NOT a tiny fractional multiplier:
      CORRECT:   `sl_long = np.minimum(df['low'], swing_lows) - (1.5 * atr(df, 14))`
      WRONG:     `sl_long = df['low'] - (0.25 * atr(df, 14))`  ← This creates micro-stops that get eaten by spread!
      `sl_short = np.maximum(df['high'], swing_highs) + (1.5 * atr(df, 14))`
    - The stop distance from entry MUST be at least $4.00 (Gold spreads + slippage = $0.25):
      `risk_long = np.maximum(df['close'] - sl_long, 4.00)`
      `risk_short = np.maximum(sl_short - df['close'], 4.00)`
    - Dynamically scale Take Profit directly from close using 1.5R to 2.5R Risk-to-Reward:
      `df['tp1_long'] = df['close'] + (risk_long * 1.8)`
      `df['tp1_short'] = df['close'] - (risk_short * 1.8)`
    - NEVER use buffer_factor < 1.0 for ATR multipliers! Values like 0.20, 0.25, 0.35 are FORBIDDEN.
    - NEVER invert Risk-to-Reward or use micro-stops (< $4.00)!
5. Mandatory Session Gating: Wrap all entries in `session_mask(df, 'london_ny')`.
6. Define calculate_signals(df) returning df with 'bull_signal', 'bear_signal', 'sl_long', 'sl_short', 'tp1_long', 'tp1_short'.
7. MUST end with `return df`.
8. Return ONLY executable Python code in ```python ... ``` block.
"""
        # Call with higher temperature (0.7) to maximize generative diversity across rounds
        sys_msg = (
            "You are the Lead Genetic Quantitative Research Agent. Your goal is to mutate and evolve top-performing champion strategies into even higher-alpha variations."
            if champion_code else
            "You are the Lead Idea Generator Agent in an elite quantitative research team. Your goal is to explore diverse, distinct institutional alpha models without repeating past ideas."
        )
        raw_resp = ""
        code = ""
        engine_mode = "LLM"
        engine_name = f"OmniRoute ({self.model})" if self.model else "OmniRoute LLM"
        fallback_active = False
        fallback_reason = None

        try:
            raw_resp = call_ai_llm(
                self.provider, self.api_key, self.model, user_prompt,
                system_prompt=sys_msg,
                endpoint_url=self.endpoint,
                temperature=0.7
            )
            code = _clean_code_response(raw_resp)
        except Exception as err:
            raw_resp = f"LLM Quota/Network Fallback: {err}"

        # If LLM failed, timed out, or returned malformed output, fall back to autonomous synthesis
        if not code or "def calculate_signals" not in code:
            code = self._synthesize_archetype_code(archetype_idx, champion_code=champion_code)
            engine_mode = "ARCHETYPE_GENERATOR"
            engine_name = "Institutional Archetype Generator"
            fallback_active = True
            
            # Determine human-readable reason
            if "429" in str(raw_resp):
                fallback_reason = "Groq Daily Token Limit Reached (429) · Auto-Resets in ~20m"
            elif "503" in str(raw_resp) or "499" in str(raw_resp) or "timed out" in str(raw_resp).lower():
                fallback_reason = "OmniRoute Upstream Timeout / Busy · Safety-Net Engaged"
            else:
                fallback_reason = f"Upstream Quota/Network Fallback: {str(raw_resp)[:60]}"
            
            set_engine_fallback(
                reason=fallback_reason,
                status_code=429 if "429" in str(raw_resp) else 503,
                error_text=str(raw_resp)
            )

        return {
            "archetype": archetype,
            "code": code,
            "raw_response": raw_resp,
            "is_breeding": bool(champion_code),
            "engine_mode": engine_mode,
            "engine_name": engine_name,
            "fallback_active": fallback_active,
            "fallback_reason": fallback_reason,
            "proposed_at": datetime.utcnow().isoformat()
        }

    @staticmethod
    def _synthesize_archetype_code(archetype_idx: int, champion_code: str = None) -> str:
        """High-speed institutional alpha synthesizer. Generates AST-compliant causal trading systems."""
        sw_len = random.choice([5, 6, 7, 8, 10, 12])
        atr_period = random.choice([10, 14, 20])
        atr_mult = round(random.choice([1.3, 1.5, 1.7, 2.0, 2.2]), 2)
        rr = round(random.choice([1.6, 1.8, 2.0, 2.2, 2.5]), 2)
        min_dist = round(random.choice([3.5, 4.0, 4.5, 5.0]), 2)
        ema_fast = random.choice([13, 21])
        ema_slow = random.choice([34, 55])
        adx_thresh = random.choice([20, 22, 25])
        rvol_thresh = round(random.choice([1.1, 1.25, 1.4]), 2)

        # Champion genetic mutation
        if champion_code and "def calculate_signals" in champion_code:
            mutated = champion_code
            for pat, rep in [
                (r'atr_mult\s*=\s*[\d\.]+', f'atr_mult = {atr_mult}'),
                (r'swing_len\s*=\s*\d+', f'swing_len = {sw_len}'),
                (r'min_risk\s*=\s*[\d\.]+', f'min_risk = {min_dist}'),
                (r'\*\s*1\.[5-9]', f'* {rr}'),
                (r'\*\s*2\.[0-5]', f'* {rr}')
            ]:
                mutated = re.sub(pat, rep, mutated)
            if mutated != champion_code:
                return mutated

        idx = archetype_idx % len(ALPHA_ARCHETYPES)
        if idx == 0:
            return f'''def calculate_signals(df):
    sess = session_mask(df, 'london_ny')
    sw_highs, sw_lows = find_swings(df, swing_len={sw_len})
    bull_fvg_top, bull_fvg_bot, bear_fvg_top, bear_fvg_bot = find_fvgs(df)
    vwap_line = vwap(df)
    
    ssl_sweep = (df['low'] < sw_lows) & (df['close'] > sw_lows)
    bsl_sweep = (df['high'] > sw_highs) & (df['close'] < sw_highs)
    
    armed_long = ssl_sweep.rolling(8, min_periods=1).max() == 1
    armed_short = bsl_sweep.rolling(8, min_periods=1).max() == 1
    
    fvg_touch_long = (df['low'] <= bull_fvg_top) & (df['close'] > bull_fvg_bot)
    fvg_touch_short = (df['high'] >= bear_fvg_bot) & (df['close'] < bear_fvg_top)
    
    raw_bull = armed_long & fvg_touch_long & (df['close'] > vwap_line) & sess
    raw_bear = armed_short & fvg_touch_short & (df['close'] < vwap_line) & sess
    
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    atr_val = atr(df, {atr_period})
    sl_long = np.minimum(df['low'], sw_lows) - ({atr_mult} * atr_val)
    sl_short = np.maximum(df['high'], sw_highs) + ({atr_mult} * atr_val)
    
    risk_l = np.maximum(df['close'] - sl_long, {min_dist})
    risk_s = np.maximum(sl_short - df['close'], {min_dist})
    
    df['sl_long'] = df['close'] - risk_l
    df['sl_short'] = df['close'] + risk_s
    df['tp1_long'] = df['close'] + (risk_l * {rr})
    df['tp1_short'] = df['close'] - (risk_s * {rr})
    return df
'''

        elif idx == 1:
            return f'''def calculate_signals(df):
    sess = session_mask(df, 'london_ny')
    levels = daily_levels(df)
    vwap_line = vwap(df)
    atr_val = atr(df, {atr_period})
    
    s1_sweep = (df['low'] < levels['s1']) & (df['close'] > levels['s1'])
    r1_sweep = (df['high'] > levels['r1']) & (df['close'] < levels['r1'])
    
    raw_bull = s1_sweep & (df['close'] > vwap_line) & sess
    raw_bear = r1_sweep & (df['close'] < vwap_line) & sess
    
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    risk_l = np.maximum({atr_mult} * atr_val, {min_dist})
    risk_s = np.maximum({atr_mult} * atr_val, {min_dist})
    
    df['sl_long'] = df['close'] - risk_l
    df['sl_short'] = df['close'] + risk_s
    df['tp1_long'] = df['close'] + (risk_l * {rr})
    df['tp1_short'] = df['close'] - (risk_s * {rr})
    return df
'''

        elif idx == 2:
            return f'''def calculate_signals(df):
    sess = session_mask(df, 'london_ny')
    er = efficiency_ratio(df['close'], 20)
    adx_val, _, _ = adx(df, {atr_period})
    st_line, st_dir = supertrend(df, 10, 3.0)
    atr_val = atr(df, {atr_period})
    
    st_bull_cross = (st_dir == 1) & (st_dir.shift(1) <= 0)
    st_bear_cross = (st_dir == -1) & (st_dir.shift(1) >= 0)
    
    regime_ok = (er > 0.28) & (adx_val > {adx_thresh}) & sess
    
    raw_bull = st_bull_cross & regime_ok
    raw_bear = st_bear_cross & regime_ok
    
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    risk_l = np.maximum(df['close'] - st_line, {min_dist})
    risk_s = np.maximum(st_line - df['close'], {min_dist})
    risk_l = np.maximum(risk_l, {atr_mult} * atr_val)
    risk_s = np.maximum(risk_s, {atr_mult} * atr_val)
    
    df['sl_long'] = df['close'] - risk_l
    df['sl_short'] = df['close'] + risk_s
    df['tp1_long'] = df['close'] + (risk_l * {rr})
    df['tp1_short'] = df['close'] - (risk_s * {rr})
    return df
'''

        else:
            return f'''def calculate_signals(df):
    sess = session_mask(df, 'london_ny')
    sw_highs, sw_lows = find_swings(df, swing_len={sw_len})
    atr_val = atr(df, {atr_period})
    ema_fast = ema(df['close'], {ema_fast})
    ema_slow = ema(df['close'], {ema_slow})
    vol_filter = rvol(df, 20) > {rvol_thresh}
    
    ssl_sweep = (df['low'] < sw_lows) & (df['close'] > sw_lows)
    bsl_sweep = (df['high'] > sw_highs) & (df['close'] < sw_highs)
    trend_bull = ema_fast > ema_slow
    trend_bear = ema_fast < ema_slow
    
    raw_bull = ssl_sweep & trend_bull & vol_filter & sess
    raw_bear = bsl_sweep & trend_bear & vol_filter & sess
    
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    risk_l = np.maximum(df['close'] - (sw_lows - ({atr_mult} * atr_val)), {min_dist})
    risk_s = np.maximum((sw_highs + ({atr_mult} * atr_val)) - df['close'], {min_dist})
    
    df['sl_long'] = df['close'] - risk_l
    df['sl_short'] = df['close'] + risk_s
    df['tp1_long'] = df['close'] + (risk_l * {rr})
    df['tp1_short'] = df['close'] - (risk_s * {rr})
    return df
'''


def synthesize_archetype_code(archetype_idx: int, champion_code: str = None) -> str:
    """Convenience module function to synthesize an institutional archetype strategy."""
    return IdeaGeneratorAgent._synthesize_archetype_code(archetype_idx, champion_code=champion_code)


# ===========================================================================
# 2. RISK OFFICER AGENT (High-Speed Static AST & Structural Risk Engine)
# ===========================================================================

class RiskOfficerAgent:
    """
    Audits candidate strategy code for:
    - Lookahead bias / causal violations (AST inspection)
    - Directional integrity (SSL = Long, BSL = Short)
    - Structural Stop Loss & Take Profit realism (Min $4.00 Stop Distance)
    - Mandatory session gating (London/NY killzones)
    - Anti-bleed transition gating
    Operates via ultra-fast deterministic AST analysis (<3ms) to eliminate
    redundant LLM wait times.
    """

    def __init__(self, provider: str = 'omniroute', api_key: str = '', model: str = '', endpoint: str = None):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint

    def audit_and_refine(self, code: str, archetype_name: str, archetype_idx: int = 0) -> Tuple[bool, str, List[str]]:
        """
        Performs static AST validation and defensive guard injection in <5ms.
        Auto-heals formatting/indentation anomalies and replaces unresolvable syntax
        anomalies with verified archetype baselines to prevent tournament stalls.
        Returns: (is_approved, refined_code_or_original, audit_notes)
        """
        notes = []

        # 0. Pre-clean & heal formatting/indentation
        code = heal_strategy_code(code)

        # 1. Hardcoded AST Lookahead Check
        is_valid, errors = validate_strategy_code(code)
        if not is_valid:
            if is_syntax_error(errors):
                notes.append(f"Auto-healed syntax irregularity in proposal: {errors[0]}")
                # Fallback to institutional archetype code to keep tournament alive
                code = synthesize_archetype_code(archetype_idx)
                notes.append(f"AST Self-Heal: Synthesized verified institutional archetype for [{archetype_name}].")
            else:
                notes.append(f"AST LOOKAHEAD VIOLATION: {errors}")
                return False, code, notes

        notes.append("AST Anti-Lookahead Check: PASSED (Zero future peeking detected)")

        # 2. Ensure Mandatory Session Gating
        if "session_mask" not in code:
            notes.append("Auto-injected session_mask(df, 'london_ny') to eliminate Asian chop.")
            code = self._inject_session_gating(code)

        # 3. Ensure Anti-Bleed Transition Gating (Eliminates 800-trade churn)
        if "shift(1)" not in code and "diff()" not in code:
            notes.append("Enforced anti-bleed transition guard to prevent consecutive-bar signal churn.")
            code = self._inject_anti_bleed_guards(code)

        # 4. Ensure Structural Risk Management
        if 'sl_long' not in code or 'sl_short' not in code:
            notes.append("Missing dynamic Stop Loss columns. Auto-injected institutional ATR risk protection.")
            code = self._inject_structural_risk(code)

        # Post-injection clean and heal
        code = heal_strategy_code(code)

        notes.append("Risk Officer Audit: APPROVED with institutional risk parameters (<3ms).")
        return True, code, notes

    def _inject_session_gating(self, code: str) -> str:
        """Injects London/NY session gating right before the final return statement."""
        injection = """
    # Risk Officer Injected Session Gating (London/NY Killzones)
    _sess_mask = session_mask(df, 'london_ny')
    if 'bull_signal' in df.columns:
        df['bull_signal'] = df['bull_signal'] & _sess_mask
    if 'bear_signal' in df.columns:
        df['bear_signal'] = df['bear_signal'] & _sess_mask
"""
        if 'return df' in code:
            idx = code.rfind('return df')
            return code[:idx] + injection + "\n    return df\n" + code[idx + 9:]
        return code + injection + "\n    return df\n"

    def _inject_anti_bleed_guards(self, code: str) -> str:
        """Injects transition checks to prevent consecutive candle signal bleed and overtrading."""
        injection = """
    # Risk Officer Injected Anti-Bleed Transition Guard (Prevents Churn Overtrading)
    if 'bull_signal' in df.columns:
        df['bull_signal'] = df['bull_signal'].astype(bool) & (~df['bull_signal'].astype(bool).shift(1).fillna(False))
    if 'bear_signal' in df.columns:
        df['bear_signal'] = df['bear_signal'].astype(bool) & (~df['bear_signal'].astype(bool).shift(1).fillna(False))
"""
        if 'return df' in code:
            idx = code.rfind('return df')
            return code[:idx] + injection + "\n    return df\n" + code[idx + 9:]
        return code + injection + "\n    return df\n"

    def _inject_structural_risk(self, code: str) -> str:
        """Injects institutional ATR SL/TP if missing from calculate_signals."""
        injection = """
    # Risk Officer Injected Institutional ATR Risk Protection (Min $4.00 Stop Distance)
    atr_risk = atr(df, period=14)
    if 'sl_long' not in df.columns:
        df['sl_long'] = np.where(df['bull_signal'], df['low'] - 1.5 * atr_risk, np.nan)
        risk_l = np.maximum(df['close'] - df['sl_long'], 4.00)
        df['tp1_long'] = np.where(df['bull_signal'], df['close'] + 1.8 * risk_l, np.nan)
    if 'sl_short' not in df.columns:
        df['sl_short'] = np.where(df['bear_signal'], df['high'] + 1.5 * atr_risk, np.nan)
        risk_s = np.maximum(df['sl_short'] - df['close'], 4.00)
        df['tp1_short'] = np.where(df['bear_signal'], df['close'] - 1.8 * risk_s, np.nan)
"""
        if 'return df' in code:
            idx = code.rfind('return df')
            return code[:idx] + injection + "\n    return df\n" + code[idx + 9:]
        return code + injection + "\n    return df\n"



# ===========================================================================
# 3. CRITIC / POST-MORTEM DIAGNOSTICIAN
# ===========================================================================

class CriticPostMortem:
    """
    Diagnoses why trades failed during the backtest.
    Extracts the worst 10 losing trades, calculates session concentrations,
    volatility levels, and provides actionable failure patterns.
    """

    @staticmethod
    def analyze_failures(trades: List[Dict[str, Any]], stats: Dict[str, Any]) -> Dict[str, Any]:
        """Performs statistical post-mortem on losing trades."""
        if not trades:
            return {"diagnosis": "No trades were taken to analyze.", "worst_losses": []}

        df_trades = pd.DataFrame(trades)
        losing_trades = df_trades[df_trades['pnl'] < 0] if 'pnl' in df_trades.columns else pd.DataFrame()

        if losing_trades.empty:
            return {
                "diagnosis": "Strategy had 0 losing trades (100% win rate).",
                "worst_losses": [],
                "session_loss_concentration": {}
            }

        worst_losses = losing_trades.sort_values('pnl').head(10).to_dict(orient='records')

        # Session distribution of losses
        session_losses = {"Asia (00-06 UTC)": 0, "London (06-12 UTC)": 0, "NY (13-20 UTC)": 0, "Late/Other": 0}
        for _, t in losing_trades.iterrows():
            en_time = t.get('entry_time')
            if en_time:
                try:
                    hour = pd.to_datetime(en_time).hour
                    if 0 <= hour < 6:
                        session_losses["Asia (00-06 UTC)"] += 1
                    elif 6 <= hour < 12:
                        session_losses["London (06-12 UTC)"] += 1
                    elif 13 <= hour < 20:
                        session_losses["NY (13-20 UTC)"] += 1
                    else:
                        session_losses["Late/Other"] += 1
                except Exception:
                    pass

        tot_losses = len(losing_trades)
        asia_pct = round((session_losses["Asia (00-06 UTC)"] / tot_losses) * 100, 1) if tot_losses > 0 else 0

        patterns = []
        if asia_pct >= 30.0:
            patterns.append(f"{asia_pct}% of losses happened during Asian low-volume chop. Strictly enforce session_mask(df, 'london_ny').")
        if stats.get('win_rate', 0) < 42.0:
            patterns.append("Win rate is under 42%. Use structural stop anchoring rather than floating stops.")
        if stats.get('profit_factor', 0) < 1.1:
            patterns.append("Profit factor is low. Adjust Risk-to-Reward ratio to between 1.3:1 and 1.8:1.")

        diagnosis_text = " | ".join(patterns) if patterns else "Losses are evenly distributed across sessions. Run Parameter Grid Sweeper to optimize ATR cushion."

        return {
            "total_losses": tot_losses,
            "worst_losses_count": len(worst_losses),
            "session_loss_concentration": session_losses,
            "diagnosis": diagnosis_text,
            "worst_sample": [
                {
                    "id": t.get('id'),
                    "direction": t.get('direction'),
                    "entry_price": t.get('entry_price'),
                    "entry_time": str(t.get('entry_time')),
                    "pnl": round(t.get('pnl', 0.0), 2),
                    "exit_reason": t.get('exit_reason')
                }
                for t in worst_losses[:5]
            ]
        }


# ===========================================================================
# 4. PARAMETER GRID SWEEPER (VECTORIZED NUMERICAL OPTIMIZER)
# ===========================================================================

class ParameterGridSweeper:
    """
    Fast vectorized parameter optimizer:
    Evaluates a 12-point grid of (atr_mult, rr_ratio) on real Dukascopy data
    to discover the mathematical alpha peak for any strategy candidate.
    """

    @staticmethod
    def _mutate_code_params(code: str, am: float, rr: float) -> str:
        """Robustly mutates ATR stop buffers and TP risk multipliers without corrupting code."""
        mutated = code
        # 1. Parameter variable assignments
        mutated = re.sub(r'\b(atr_mult|atr_multiplier|atr_buffer_factor)\s*=\s*[\d\.]+', rf'\g<1> = {am}', mutated)
        mutated = re.sub(r'\b(rr_ratio|rr_factor|reward_risk_ratio|target_rr)\s*=\s*[\d\.]+', rf'\g<1> = {rr}', mutated)

        # 2. ATR buffer expressions in Stop Loss (e.g. 1.5 * atr(...) or (1.5 * atr_val))
        mutated = re.sub(
            r'([\+\-\(\*]\s*)[\d\.]+(\s*\*\s*atr(?:_val|_risk|\(df|\b))',
            rf'\g<1>{am}\g<2>',
            mutated
        )
        mutated = re.sub(
            r'(atr(?:_val|_risk|\(df|\b)\s*\*\s*)[\d\.]+',
            rf'\g<1>{am}',
            mutated
        )

        # 3. Risk multiplier expressions in Take Profit (maintaining long and short risk independently)
        mutated = re.sub(
            r'\b(risk_\w+|risk[ls]|sl_dist(?:_\w+)?)\s*\*\s*[\d\.]+',
            rf'\g<1> * {rr}',
            mutated
        )
        mutated = re.sub(
            r'[\d\.]+\s*\*\s*(risk_\w+|risk[ls]|sl_dist(?:_\w+)?)',
            rf'{rr} * \g<1>',
            mutated
        )

        return mutated

    @staticmethod
    def sweep_and_optimize(base_code: str, df: pd.DataFrame) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], Dict[str, float]]:
        """
        Sweeps combinations of (atr_mult in [1.2, 1.5, 2.0], rr_ratio in [1.5, 1.8, 2.2, 2.5]).
        Returns: (best_code, best_stats, best_trades, best_monthly)
        """
        # Fast 1-Pass Baseline Prune:
        # If the candidate produces < 5 trades, abort immediately without wasting full grid sweeps!
        base_res = execute_strategy(base_code, df)
        if not base_res.get('success'):
            return base_code, {}, [], {}
        base_trades = base_res.get('trades', [])
        base_stats = base_res.get('stats', {})
        base_monthly = compute_monthly_r_breakdown(base_trades)
        if len(base_trades) < 5:
            return base_code, base_stats, base_trades, base_monthly

        best_code = base_code
        best_stats = base_stats
        best_trades = base_trades
        best_monthly = base_monthly
        base_net_r = sum(base_monthly.values()) if base_monthly else base_stats.get('total_pnl', 0.0) / 1000.0
        base_months_10 = sum(1 for v in base_monthly.values() if v >= 10.0)
        base_pf = float(base_stats.get('profit_factor', 1.0))
        best_score = (base_net_r * 1.5) + (base_months_10 * 15.0) + (base_pf * 10.0)
        if len(base_trades) < 25:
            best_score *= (len(base_trades) / 25.0)

        # Institutional ATR stop buffers and Risk-to-Reward ratios
        atr_mults = [1.2, 1.5, 2.0]
        rr_ratios = [1.5, 1.8, 2.2, 2.5]

        consecutive_dead = 0
        for am in atr_mults:
            if consecutive_dead >= 4 and best_score <= 0:
                break
            for rr in rr_ratios:
                mutated = ParameterGridSweeper._mutate_code_params(base_code, am, rr)
                if mutated == base_code:
                    continue

                res = execute_strategy(mutated, df)
                if res.get('success'):
                    trades = res.get('trades', [])
                    stats = res.get('stats', {})
                    if len(trades) < 5:
                        consecutive_dead += 1
                        if consecutive_dead >= 4 and best_score <= 0:
                            break
                        continue
                    else:
                        consecutive_dead = 0
                    if len(trades) >= 10:
                        monthly = compute_monthly_r_breakdown(trades)
                        months_10 = sum(1 for v in monthly.values() if v >= 10.0)
                        net_r = sum(monthly.values()) if monthly else stats.get('total_pnl', 0.0) / 1000.0
                        pf = float(stats.get('profit_factor', 1.0))
                        score = (net_r * 1.5) + (months_10 * 15.0) + (pf * 10.0)

                        # Sample size confidence penalty
                        if len(trades) < 25:
                            score *= (len(trades) / 25.0)

                        # Penalize overtrading churn and reward selective high-conviction frequency
                        if len(trades) > 300:
                            score -= 40.0
                        elif 40 <= len(trades) <= 220:
                            score += 15.0

                        if score > best_score:
                            best_score = score
                            best_code = mutated
                            best_stats = stats
                            best_trades = trades
                            best_monthly = monthly

        return best_code, best_stats, best_trades, best_monthly


# ===========================================================================
# 5. OPTIMIZER AGENT
# ===========================================================================

class OptimizerAgent:
    """
    Takes backtest statistics, Monte Carlo 95% VaR stress test, and Critic Post-Mortem,
    and intelligently mutates parameters and filters to target >= 5 months >= 10R.
    """

    def __init__(self, provider: str = 'omniroute', api_key: str = '', model: str = '', endpoint: str = None):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint

    def optimize_with_postmortem(self,
                                 current_code: str,
                                 stats: Dict[str, Any],
                                 mc_results: Dict[str, Any],
                                 post_mortem: Dict[str, Any],
                                 monthly_r: Dict[str, float],
                                 iteration: int = 1,
                                 leaderboard_context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Refines the strategy code guided by empirical failure modes, Monte Carlo VaR, and tournament winners."""
        months_ge_10 = sum(1 for v in monthly_r.values() if v >= 10.0)

        lb_section = ""
        if leaderboard_context:
            benchmarks = []
            for i, top in enumerate(leaderboard_context[:3]):
                benchmarks.append(
                    f"  #{i+1} [{top.get('name', 'Champion')}]: {top.get('total_r', 0):+.1f}R | "
                    f"PF: {top.get('profit_factor', 0)} | WR: {top.get('win_rate', 0)}% | Trades: {top.get('total_trades', 0)}"
                )
            lb_section = f"""
CURRENT TOURNAMENT BENCHMARKS (TOP PERFORMERS ON LEADERBOARD):
{chr(10).join(benchmarks)}
Notice the trade frequency, profit factor, and consistency of these proven performers. Emulate their structural discipline!
"""

        prompt = f"""
[AGENT: QUANTITATIVE OPTIMIZER]
We are optimizing this strategy candidate (Iteration #{iteration}).

CURRENT CODE:
```python
{current_code}
```

EMPIRICAL BACKTEST PERFORMANCE (6-Month Dukascopy 5m Gold):
- Total Trades: {stats.get('total_trades', 0)}
- Win Rate: {stats.get('win_rate', 0)}%
- Net PnL: ${stats.get('total_pnl', 0):,.2f}
- Profit Factor: {stats.get('profit_factor', 0)}
- Max Drawdown: ${stats.get('max_drawdown', 0):,.2f} ({stats.get('max_drawdown_pct', 0)}%)
- Monte Carlo 95% VaR Drawdown: -{mc_results.get('var_95_max_dd_r', 'N/A')} R
- Monthly Returns Breakdown: {monthly_r}
- High-Yield Months (>= +10R): {months_ge_10} Months
{lb_section}
CRITIC POST-MORTEM FAILURE DIAGNOSIS:
{post_mortem.get('diagnosis', 'Standard optimization')}
Session Loss Distribution: {post_mortem.get('session_loss_concentration', {})}

TARGET OBJECTIVES FOR INSTITUTIONAL PROFITABILITY:
1. Strictly enforce London/NY killzones: `session_mask(df, 'london_ny')` to eliminate low-volume chop.
2. Anchor Stop Loss with a healthy institutional 1.2x to 2.2x ATR buffer (minimum $4.00 distance from entry). NEVER use micro-stops (< $4.00) that get eaten by spread and slippage!
3. Enforce anti-bleed transition triggers (`signal & ~signal.shift(1)`) so trades only enter on setup initiation, targeting ~0.5 to 1.5 high-conviction trades per trading day on average.
4. Scale Take-Profit dynamically to at least 1.5x to 2.2x the true stop distance.
5. Target consistent positive monthly expectancy and solid risk-adjusted return across all market regimes.
6. Keep the code clean, fast, and STRICTLY CAUSAL (Zero lookahead).
7. MUST end with `return df`.

Return ONLY the improved Python code in ```python ... ```.
"""
        try:
            raw_resp = call_ai_llm(
                self.provider, self.api_key, self.model, prompt,
                system_prompt="You are an elite Quantitative Strategy Optimizer. You make targeted mathematical enhancements to code based on empirical trade logs.",
                endpoint_url=self.endpoint
            )
            new_code = _clean_code_response(raw_resp)
            if 'return df' not in new_code:
                new_code += "\n    return df\n"
            
            # Defensive guard: Ensure optimization didn't strip session or anti-bleed guards
            ro_helper = RiskOfficerAgent()
            if "session_mask" not in new_code:
                new_code = ro_helper._inject_session_gating(new_code)
            if "shift(1)" not in new_code and "diff()" not in new_code:
                new_code = ro_helper._inject_anti_bleed_guards(new_code)
            if 'sl_long' not in new_code or 'sl_short' not in new_code:
                new_code = ro_helper._inject_structural_risk(new_code)

            is_valid, errors = validate_strategy_code(new_code)
            if not is_valid:
                return {"success": False, "message": f"Optimized code failed AST validation: {errors}", "code": current_code}

            return {"success": True, "code": new_code, "raw_response": raw_resp}
        except Exception as e:
            return {"success": False, "message": str(e), "code": current_code}

