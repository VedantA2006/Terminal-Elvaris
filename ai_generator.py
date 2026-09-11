"""
AI Strategy Generator & Continuous Autonomous Optimizer.

Supports:
  - Google Gemini, OpenAI, Anthropic Claude, Groq, OpenRouter
  - Hardcoded unbypassable Anti-Lookahead Bias System Prompt & Guardrails
  - Continuous Testing / Evolutionary Parameter Optimization
"""

import os
import re
import json
import requests
import time
from typing import Dict, Any, List, Optional
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime

from lookahead_guard import validate_strategy_code, heal_strategy_code
from strategy_executor import execute_strategy

# ---------------------------------------------------------------------------
# LIVE ENGINE TELEMETRY & FALLBACK TRACKER
# ---------------------------------------------------------------------------
ENGINE_TELEMETRY: Dict[str, Any] = {
    "mode": "IDLE",
    "provider": "omniroute",
    "model": "agentrouter/gpt-6-astra",
    "endpoint": "http://localhost:20128/v1",
    "status": "idle",
    "status_code": None,
    "last_error": None,
    "fallback_active": False,
    "fallback_reason": None,
    "active_display": "OmniRoute LLM (agentrouter/gpt-6-astra)",
    "last_updated": datetime.utcnow().strftime("%H:%M:%S")
}


def get_engine_telemetry() -> Dict[str, Any]:
    """Returns a snapshot of the active generation engine telemetry."""
    return dict(ENGINE_TELEMETRY)


def update_engine_telemetry(**kwargs):
    """Updates live generation telemetry."""
    ENGINE_TELEMETRY.update(kwargs)
    ENGINE_TELEMETRY["last_updated"] = datetime.utcnow().strftime("%H:%M:%S")


def set_engine_fallback(reason: str, status_code: int = 429, error_text: str = ""):
    """Dispatches fallback event to institutional archetype engine."""
    clean_reason = reason if reason else "Upstream rate limit cooldown"
    ENGINE_TELEMETRY.update({
        "mode": "ARCHETYPE_GENERATOR",
        "status": "fallback",
        "status_code": status_code,
        "last_error": error_text or clean_reason,
        "fallback_active": True,
        "fallback_reason": clean_reason,
        "active_display": "🟡 Institutional Archetype Generator (Fallback Active)",
        "last_updated": datetime.utcnow().strftime("%H:%M:%S")
    })


# ---------------------------------------------------------------------------
# HARDCODED ANTI-LOOKAHEAD BIAS SYSTEM PROMPT & CONCEPT INVENTORY
# ---------------------------------------------------------------------------
HARDCODED_SYSTEM_PROMPT = """
You are an elite quantitative hedge fund researcher and algorithmic trading engineer.
Your task is to write high-performing, robust Python trading strategies for XAUUSD (Gold) on 5-minute data.

🚨 CRITICAL MANDATORY RULES — STRICT CAUSALITY & ZERO LOOKAHEAD BIAS:
1. NEVER peek into the future under any circumstances.
2. PROHIBITED CONSTRUCTS:
   - Absolutely NO negative shifts: `df.shift(-n)`, `shift(periods=-n)` are STRICTLY BANNED.
   - Absolutely NO centered rolling windows: `center=True` in `.rolling()` is STRICTLY BANNED.
   - Absolutely NO backward filling: `.bfill()`, `fillna(method='bfill')` are STRICTLY BANNED.
   - Absolutely NO forward indexing into future bars: `iloc[i + n]`, `loc[...]` are STRICTLY BANNED.
   - Every calculation for bar [t] can ONLY use historical data <= [t].
3. HARDCODED SERVER VALIDATION:
   All strategy code is statically parsed by an AST Lookahead Bias Guard. Any violation will immediately reject your code and fail the test.

QUANTITATIVE RESEARCH CONCEPT INVENTORY (BUILD & EXPERIMENT USING THESE BLOCKS):
You draw from 7 proven institutional quantitative domains to discover and optimize strategies:

1. LIQUIDITY & MARKET STRUCTURE (SMC / ICT):
   - Liquidity Sweeps: BSL (Buy-Side Liquidity) sweep (high > swing_high and close < swing_high) & SSL (Sell-Side Liquidity) sweep (low < swing_low and close > swing_low).
   - Equal Highs / Equal Lows (EQH / EQL) liquidity pools.
   - Fair Value Gaps (FVG): Bullish (low[i] > high[i-2]) and Bearish (high[i] < low[i-2]), with mitigation at midpoint ((top + bot) / 2).
   - Inverted Fair Value Gaps (IFVG): Broken FVGs flipped into support/resistance.
   - Order Blocks (OB) & Displacement candles (body/range ratio > 70%).

2. MULTI-TIMEFRAME MACRO & DAILY KEY LEVELS:
   - Daily Levels: Previous Day High (PDH), Previous Day Low (PDL), Previous Day Close (PDC).
   - Floor Pivot Points: Central Pivot (PP), Resistance (R1, R2), Support (S1, S2).
   - Developing Intraday VWAP (Volume-Weighted Average Price) resetting daily.

3. STATISTICAL MECHANICS & QUANTITATIVE ARBITRAGE:
   - Rolling Z-Score: Statistical standard deviations from rolling mean ((price - mean) / std) for mean-reversion fades.
   - Vectorized Linear Regression Slope: Quantitative price drift velocity.
   - Kaufman Efficiency Ratio: Signal-to-noise filter (ER > 0.35 for trending, ER < 0.20 for ranging chop).

4. VOLATILITY REGIMES & ADAPTIVE BANDS:
   - Supertrend: Adaptive trend tracking line and regime direction (+1 bull / -1 bear).
   - Keltner Channels & Bollinger Bands: Volatility squeezes (Bollinger inside Keltner) followed by explosive directional expansion.
   - Donchian Channels: Turtle Breakout dynamics (highest high / lowest low breakout).
   - Chandelier Exit: Dynamic trailing ATR stops.

5. MOMENTUM & DIRECTIONAL DYNAMICS:
   - ADX & Directional Movement (+DI / -DI): Trend persistence filter (ADX > 25) vs consolidation (ADX < 20).
   - Wilder's RSI (14) & Stochastic: Dynamic overbought/oversold and midline transitions.
   - MACD / Histogram: Momentum acceleration and zero-line crossovers.

6. INSTITUTIONAL VOLUME FOOTPRINT:
   - RVOL (Relative Volume): Detecting institutional volume spikes (> 1.5x rolling average).
   - Volume exhaustion vs volume expansion candles.

7. SESSION TIMING & KILLZONES:
   - London Open Killzone: 06:00 - 11:00 UTC (07:00 - 12:00 CET).
   - New York Killzone: 17:30 - 24:00 UTC (13:30 - 20:00 EDT).
   - Asian Range Sweeps (Judas Swings at London Open).

API SPECIFICATION — CAUSAL BUILT-IN HELPERS:
The strategy code operates on a pandas DataFrame `df` with columns: ['open', 'high', 'low', 'close', 'volume'] and DatetimeIndex `df.index`.
You have access to: `pd`, `np`, `math`, and built-in helper functions:
- `find_swings(df, swing_len=7)` -> returns (swing_highs, swing_lows) forward-filled with active swing levels:
  - Bullish / Long Setup (SSL Sweep): `(df['low'] < swing_lows) & (df['close'] > swing_lows)` -> price sweeps below swing low and closes back above it.
  - Bearish / Short Setup (BSL Sweep): `(df['high'] > swing_highs) & (df['close'] < swing_highs)` -> price sweeps above swing high and closes back below it.
- `find_fvgs(df)` -> returns (bull_fvg_top, bull_fvg_bot, bear_fvg_top, bear_fvg_bot) forward-filled with most recent FVG boundaries.
- `daily_levels(df)` -> returns DataFrame with ['pdh', 'pdl', 'pdc', 'pp', 'r1', 's1', 'r2', 's2']
- `vwap(df)` -> returns pd.Series of daily resetting VWAP
- `session_mask(df, 'london' | 'ny' | 'asia' | 'london_ny')` -> returns boolean pd.Series
- `supertrend(df, period=10, mult=3.0)` -> returns (supertrend_line, direction [+1/-1])
- `adx(df, period=14)` -> returns (adx, p_di, m_di)
- `zscore(series, period=20)` -> returns pd.Series
- `linear_regression_slope(series, period=20)` -> returns pd.Series
- `efficiency_ratio(series, period=20)` -> returns pd.Series
- `rvol(df, period=20)` -> returns pd.Series
- `keltner_channels(df, ema_period=20, atr_period=10, mult=2.0)` -> returns (upper, lower, mid)
- `donchian_channels(df, period=20)` -> returns (upper, lower, mid)
- `bollinger_bands(series, period=20, mult=2.0)` -> returns (upper, lower, basis)
- `chandelier_exit(df, period=22, mult=3.0)` -> returns (long_stop, short_stop)
- `rsi(series, period=14)` -> returns pd.Series
- `stochastic(df, k_period=14, d_period=3)` -> returns (k, d)
- `macd(series, fast=12, slow=26, signal=9)` -> returns (macd, signal, hist)
- `sma(series, period)`, `ema(series, period)`, `smma(series, period)`, `atr(df, period)`

CRITICAL INSTITUTIONAL TRADE EXECUTION & PROFITABILITY RULES:
1. WIDE MULTI-CONFLUENCE THINKING:
   - Combine at least 2 or 3 independent quantitative pillars:
     a) Macro Context / Trend: Intraday VWAP (`df['close'] > vwap(df)`), Daily Floor Pivots (`daily_levels(df)`: S1/R1/PDH/PDL), or EMA 21/55 alignment.
     b) Execution Trigger: SMC liquidity sweeps (`find_swings`), order block retests, or volume displacement.
     c) Momentum & Volume Filter: `rvol(df, 20) > 1.2`, ADX > 22, or Kaufman Efficiency Ratio > 0.28.
     d) Mandatory Killzone Gating: Strictly execute inside high-liquidity sessions via `session_mask(df, 'london_ny')`.

2. ANTI-BLEED / NON-REPEATING TRANSITION SIGNALS (PREVENT OVERTRADING CHURN):
   - Signals MUST trigger ONLY on the FIRST candle transition of a setup:
     `raw_bull = setup_condition & sess`
     `bull_signal = raw_bull & (~raw_bull.shift(1).fillna(False))`
   - NEVER let signals fire repeatedly on consecutive bars during a rolling window!
   - Target a selective trade frequency (0.5 to 1.5 trades/day) of high-conviction entries.

3. INSTITUTIONAL RISK MANAGEMENT & MINIMUM STOP DISTANCE:
   - ALL trade entries are taken strictly at the CANDLE CLOSE (`df['close']`).
   - Stop Loss MUST have a healthy institutional buffer to survive Dukascopy spread ($0.20) and slippage ($0.05):
     Use a 1.2x to 2.2x ATR buffer:
     `sl_long = np.minimum(df['low'], swing_lows) - (1.5 * atr(df, 14))`
     `sl_short = np.maximum(df['high'], swing_highs) + (1.5 * atr(df, 14))`
   - Ensure risk distance from candle close is at least $2.50:
     `risk_long = np.maximum(df['close'] - sl_long, 2.50)`
     `risk_short = np.maximum(sl_short - df['close'], 2.50)`
   - Dynamically scale Take Profit directly from the candle close using 1.5R to 2.5R Risk-to-Reward:
     `df['tp1_long'] = df['close'] + (risk_long * 1.5)`
     `df['tp1_short'] = df['close'] - (risk_short * 1.5)`
   - NEVER invert Risk-to-Reward or use micro-stops (< $2.00)!

REQUIRED OUTPUTS:
Your code MUST define `calculate_signals(df)` and set boolean signal columns on `df`:
- `df['bull_signal']`: True when a buy/long condition triggers on bar close
- `df['bear_signal']`: True when a sell/short condition triggers on bar close
Risk management columns (calculated strictly relative to candle close):
- `df['sl_long']`: Float Stop Loss price level for long trades
- `df['sl_short']`: Float Stop Loss price level for short trades
- `df['tp1_long']`: Float Take Profit price level derived from close + (risk * RR)
- `df['tp1_short']`: Float Take Profit price level derived from close - (risk * RR)

FORMATTING:
CRITICAL: Do NOT output <think> tags, conversational commentary, or internal reasoning.
Start immediately with ```python and return ONLY clean, valid, executable Python code inside a ```python ... ``` block.
"""


def _clean_code_response(text: str) -> str:
    """Extract python code from LLM markdown response, stripping think tags and auto-healing syntax anomalies."""
    if not text:
        return ""

    raw_code = ""
    # 1. First, search for standard fenced code blocks containing calculate_signals
    fenced_blocks = re.findall(r'```(?:python)?\s*([\s\S]*?)(?:```|$)', text, re.IGNORECASE)
    for block in fenced_blocks:
        if 'def calculate_signals' in block:
            raw_code = block.strip()
            break

    if not raw_code:
        # 2. Strip closed <think>...</think> blocks
        cleaned = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)

        # 3. If def calculate_signals is present anywhere in cleaned or raw text, slice from it
        target = cleaned if 'def calculate_signals' in cleaned else text
        if 'def calculate_signals' in target:
            start_idx = target.find('def calculate_signals')
            code_part = target[start_idx:]
            # If there's an ending backtick or markdown fence, strip it
            fence_end = re.search(r'```', code_part)
            if fence_end:
                code_part = code_part[:fence_end.start()]
            raw_code = code_part.strip()
        elif fenced_blocks and fenced_blocks[0].strip():
            raw_code = fenced_blocks[0].strip()
        else:
            if '<think>' in cleaned.lower():
                cleaned = re.sub(r'<think>[\s\S]*$', '', cleaned, flags=re.IGNORECASE)
            raw_code = cleaned.strip()

    # Pass through robust AST self-healer to normalize indentation and repair truncated expressions
    return heal_strategy_code(raw_code)


def call_ai_llm(provider: str, api_key: str, model: str, prompt: str, system_prompt: str = None, endpoint_url: str = None, temperature: float = 0.3) -> str:
    """
    Calls specified LLM provider with the hardcoded anti-lookahead system prompt.
    Supports OmniRoute, OpenAI, Google Gemini, Anthropic Claude, Groq, and OpenRouter.
    """
    no_think_rule = "\n\nCRITICAL OUTPUT RULE: Output valid executable Python code inside ```python ... ``` immediately. Do NOT output <think> tags, chain-of-thought, or conversational filler."
    sys_prompt = HARDCODED_SYSTEM_PROMPT + no_think_rule
    if system_prompt:
        sys_prompt = HARDCODED_SYSTEM_PROMPT + "\n\nUser Strategy Objectives:\n" + system_prompt + no_think_rule

    provider = (provider or 'omniroute').lower()

    if provider == 'gemini':
        # Google Gemini API
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model or 'gemini-2.0-flash'}:generateContent?key={api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": sys_prompt}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": float(temperature)}
        }
        res = requests.post(endpoint, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data['candidates'][0]['content']['parts'][0]['text']

    elif provider in ('omniroute', 'openai', 'groq', 'openrouter') or endpoint_url:
        url_map = {
            'omniroute': endpoint_url or 'http://localhost:20128/v1/chat/completions',
            'openai': 'https://api.openai.com/v1/chat/completions',
            'groq': 'https://api.groq.com/openai/v1/chat/completions',
            'openrouter': 'https://openrouter.ai/api/v1/chat/completions',
        }
        url = endpoint_url if endpoint_url else url_map.get(provider, 'http://localhost:20128/v1/chat/completions')
        if not url.endswith('/chat/completions'):
            url = url.rstrip('/') + '/chat/completions'

        default_models = {
            'omniroute': 'agentrouter/gpt-6-astra',
            'openai': 'gpt-4o-mini',
            'groq': 'openai/gpt-oss-120b',
            'openrouter': 'openai/gpt-4o-mini',
        }
        if provider == 'groq':
            active_key = api_key if api_key else os.environ.get('GROQ_API_KEY', '')
            if not active_key:
                active_key = os.environ.get('OMNIROUTE_API_KEY', '')
        else:
            active_key = api_key if api_key else os.environ.get('OMNIROUTE_API_KEY', '')
        if not active_key and provider in ('omniroute', 'groq'):
            raise ValueError(f"{provider.upper()}_API_KEY environment variable is not set. See .env.example.")

        headers = {
            'Authorization': f"Bearer {active_key}",
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
        }

        # Resilient candidate model fallback if primary model hits 429/503 quota limits
        primary_model = model or default_models.get(provider, 'agentrouter/gpt-6-astra')
        if provider == 'omniroute':
            if primary_model in ('auto/best-coding', 'auto/best-reasoning', 'auto', 'groq/qwen/qwen3.6-27b', 'qwen/qwen3.6-27b'):
                primary_model = 'agentrouter/gpt-6-astra'
            # Dedicated: agentrouter/gpt-6-astra is candidate #1 until full limit (429) is hit
            raw_candidates = [
                'agentrouter/gpt-6-astra',
                'groq/openai/gpt-oss-120b',
                'mistral/codestral-latest',
                'groq/qwen/qwen3.8-27b',
                'groq/openai/gpt-oss-20b'
            ]
            if primary_model not in raw_candidates:
                raw_candidates.insert(0, primary_model)
        elif provider == 'groq':
            if primary_model in ('qwen/qwen3.6-27b', 'auto'):
                primary_model = 'openai/gpt-oss-120b'
            raw_candidates = [
                primary_model,
                'openai/gpt-oss-120b',
                'qwen/qwen3.8-27b',
                'openai/gpt-oss-20b'
            ]
        else:
            raw_candidates = [primary_model]

        # Deduplicate while preserving order
        seen = set()
        candidates = []
        for c in raw_candidates:
            if c not in seen:
                seen.add(c)
                candidates.append(c)

        last_err = None
        for cand_model in candidates:
            payload = {
                'model': cand_model,
                'messages': [
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                'temperature': float(temperature),
                'max_tokens': 1600,
            }
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=(10, 120))
                if res.status_code in (400, 401, 404, 429, 500, 502, 503, 504):
                    last_err = requests.HTTPError(f"HTTP {res.status_code} on model {cand_model}: {res.text[:120]}")
                    time.sleep(1.0)
                    continue
                res.raise_for_status()
                data = res.json()
                if data.get('choices') and data['choices'][0].get('message'):
                    msg = data['choices'][0]['message']
                    content = msg.get('content', '') or ''
                    # Some reasoning models might put generated code in reasoning or thought fields
                    if not content.strip() and msg.get('reasoning'):
                        content = msg.get('reasoning', '')
                    if content.strip():
                        update_engine_telemetry(
                            mode="LLM",
                            provider=provider,
                            model=cand_model,
                            endpoint=url,
                            status="online",
                            status_code=200,
                            last_error=None,
                            fallback_active=False,
                            fallback_reason=None,
                            active_display=f"🟢 Live LLM: {cand_model} (200 OK)"
                        )
                        return content
            except Exception as e:
                last_err = e
                time.sleep(1.0)
                continue

        # If OmniRoute failed completely, but GROQ_API_KEY is available in env, seamlessly try Groq direct
        if provider == 'omniroute' and os.environ.get('GROQ_API_KEY'):
            try:
                return call_ai_llm(
                    provider='groq',
                    api_key=os.environ.get('GROQ_API_KEY'),
                    model='openai/gpt-oss-120b',
                    prompt=prompt,
                    system_prompt=system_prompt,
                    endpoint_url='https://api.groq.com/openai/v1',
                    temperature=temperature
                )
            except Exception:
                pass

        err_msg = str(last_err) if last_err else "All candidate models failed or rate limited"
        sc = 429 if "429" in err_msg else 503
        set_engine_fallback(
            reason=f"Upstream Rate Limit (Groq/OmniRoute {sc}) · Daily Quota Cooldown",
            status_code=sc,
            error_text=err_msg
        )
        if last_err:
            raise last_err
        raise RuntimeError("No response returned from AI provider.")

    elif provider == 'anthropic':
        # Anthropic Claude API
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': model or 'claude-3-5-sonnet-20241022',
            'max_tokens': 4000,
            'system': sys_prompt,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': float(temperature),
        }
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data['content'][0]['text']

    else:
        raise ValueError(f"Unsupported AI provider: {provider}")


def generate_strategy_code(provider: str, api_key: str, model: str, user_prompt: str, endpoint_url: str = None) -> Dict[str, Any]:
    """
    Generates strategy code, runs lookahead validation, and returns clean code.
    """
    if not api_key:
        if provider == 'omniroute':
            api_key = os.environ.get('OMNIROUTE_API_KEY', '')
            if not api_key:
                return {'success': False, 'message': 'OMNIROUTE_API_KEY environment variable is not set. See .env.example.'}
        else:
            return {'success': False, 'message': 'API Key is required to call AI provider.'}

    full_prompt = f"""
Create a high-win-rate, robust Python algorithmic trading strategy for XAUUSD (Gold) on 5-minute candles.
User Request / Strategy Concept:
{user_prompt}

Remember:
- Do NOT use lookahead bias (no df.shift(-n), no center=True, no bfill).
- Calculate df['bull_signal'], df['bear_signal'], and realistic df['sl_long'], df['sl_short'], df['tp_long'], df['tp_short'].
- Return ONLY the executable Python code inside a ```python ``` block.
"""
    try:
        raw_response = call_ai_llm(provider, api_key, model, full_prompt, endpoint_url=endpoint_url)
        code = _clean_code_response(raw_response)

        # Validate Lookahead
        is_valid, errors = validate_strategy_code(code)
        if not is_valid:
            return {
                'success': False,
                'error_type': 'LOOKAHEAD_BIAS',
                'message': 'AI generated code contained lookahead bias and was blocked.',
                'errors': errors,
                'raw_code': code,
            }

        return {
            'success': True,
            'code': code,
            'lookahead_validated': True,
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}


def optimize_strategy_step(provider: str, api_key: str, model: str, current_code: str,
                           previous_stats: Dict[str, Any], iteration: int,
                           df: pd.DataFrame, endpoint_url: str = None) -> Dict[str, Any]:
    """
    Autonomous optimization step: Takes previous backtest results and asks AI to
    refine parameters or add causal filters to improve Profit Factor and Sharpe Ratio.
    """
    prompt = f"""
We are running autonomous continuous optimization iteration #{iteration}.
Here is the current strategy code:
```python
{current_code}
```

Previous Backtest Performance on 5-min Dukascopy Gold data:
- Total Trades: {previous_stats.get('total_trades', 0)}
- Win Rate: {previous_stats.get('win_rate', 0)}%
- Net Profit: ${previous_stats.get('total_pnl', 0):,.2f}
- Profit Factor: {previous_stats.get('profit_factor', 0)}
- Max Drawdown: ${previous_stats.get('max_drawdown', 0):,.2f} ({previous_stats.get('max_drawdown_pct', 0)}%)
- Sharpe Ratio: {previous_stats.get('sharpe_ratio', 0)}

OBJECTIVE:
Mutate and optimize the strategy to improve the Profit Factor, increase Win Rate, and reduce Max Drawdown.
You may adjust indicator periods, thresholds, add trend filters (e.g. higher-timeframe EMA, ADX, or RSI regime filter), or optimize SL/TP distance.

MANDATORY:
- STRICT ZERO LOOKAHEAD BIAS: No `shift(-n)`, no `center=True`, no `bfill()`.
- Return ONLY the improved Python code in a ```python ... ``` block.
"""
    try:
        raw_response = call_ai_llm(provider, api_key, model, prompt, endpoint_url=endpoint_url)
        new_code = _clean_code_response(raw_response)

        # 1. AST Lookahead check
        is_valid, errors = validate_strategy_code(new_code)
        if not is_valid:
            return {
                'success': False,
                'error_type': 'LOOKAHEAD_BIAS',
                'message': 'Mutated code failed AST Lookahead check.',
                'errors': errors,
            }

        # 2. Run backtest directly to evaluate
        exec_res = execute_strategy(new_code, df)
        if not exec_res['success']:
            return {
                'success': False,
                'error_type': 'EXECUTION_ERROR',
                'message': exec_res.get('message', 'Execution failed'),
            }

        new_stats = exec_res['stats']
        trades_count = len(exec_res['trades'])

        # Score strategy: weighted combination of Profit Factor, Sharpe, and Drawdown
        pf = min(new_stats.get('profit_factor', 0), 10.0)
        sharpe = new_stats.get('sharpe_ratio', 0)
        dd_pct = new_stats.get('max_drawdown_pct', 0)
        
        # Base quantitative score
        score = (pf * 2.5) + (sharpe * 1.5) - (dd_pct * 0.15)
        
        # Statistical robustness guard: heavily penalize tiny sample sizes (< 15 trades over 6 months)
        if trades_count < 15:
            score -= 10.0  # Reject curve-fitting on 2-3 lucky trades
        elif trades_count >= 50:
            score += 1.0   # Bonus for statistically significant sample size

        return {
            'success': True,
            'iteration': iteration,
            'code': new_code,
            'stats': new_stats,
            'score': round(score, 2),
            'trades_count': trades_count,
            'statistically_significant': trades_count >= 20,
            'exec_result': exec_res,
        }

    except Exception as e:
        return {'success': False, 'message': str(e)}
