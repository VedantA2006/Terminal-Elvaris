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

def load_archetypes():
    import json
    import os
    path = os.path.join("data", "archetypes_XAUUSD.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return []

def save_archetypes(archetypes):
    import json
    import os
    path = os.path.join("data", "archetypes_XAUUSD.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(archetypes, f, indent=4)



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
        self.model = model or 'groq/llama-3.1-70b-versatile'
        if self.model in ('auto/best-coding', 'auto/best-reasoning', 'auto', 'groq/qwen/qwen3.6-27b', 'qwen/qwen3.6-27b', 'agentrouter/gpt-6-astra', 'mistral/codestral-latest'):
            self.model = 'groq/llama-3.1-70b-versatile'
        self.endpoint = endpoint or os.environ.get('OMNIROUTE_ENDPOINT', 'http://localhost:20128/v1')

    def propose_strategy(self, archetype_idx: int = 0, recent_hypotheses: List[str] = None, custom_focus: str = None,
                         champion_code: str = None, second_parent_code: str = None,
                         failure_memory: List[str] = None, leaderboard_code_context: List[Dict] = None,
                         negative_constraints: List[str] = None, temperature: float = 0.7) -> Dict[str, Any]:
        """Generates an initial strategy code proposal based on a chosen archetype or breeds a champion parent."""
        archetype = load_archetypes()[archetype_idx % len(load_archetypes())]

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

        # Pre-LLM Negative Parameter Cache to prevent duplicate signal footprints
        negative_section = ""
        if negative_constraints:
            formatted_negatives = "\n".join(f"- {c}" for c in negative_constraints[-8:])
            negative_section = f"""
🚫 NEGATIVE PARAMETER CACHE (DO NOT DUPLICATE RECENT PARAMETERS):
The following parameter combinations were RECENTLY TESTED for this archetype:
{formatted_negatives}

CRITICAL MANDATE: You MUST use DIFFERENT lookbacks, periods, or multipliers! For example, if recent runs used swing_len=7 and period=20, you must explore swing_len in [10, 14, 21], ema in [34, 89], or wider ATR multipliers in [1.8, 2.4, 3.0]. Do NOT replicate the same parameter footprint!
"""

        # Anti-Mode-Collapse Mandate: Forbid ssl_sweep on non-sweep archetypes
        arch_name_lower = archetype['name'].lower()
        is_sweep_archetype = any(k in arch_name_lower for k in ['sweep', 'liquidity', 'judas', 'fvg', 'smc', 'asian range'])
        purity_mandate = ""
        if not is_sweep_archetype and not champion_code:
            purity_mandate = f"""
🚫 CRITICAL ANTI-MODE-COLLAPSE MANDATE (ARCHETYPE PURITY):
You are generating code specifically for the [{archetype['name']}] archetype.
DO NOT use swing sweeps (ssl_sweep / bsl_sweep) or `find_swings(...)` sweeps as your primary entry triggers!
You MUST implement the authentic mathematical indicators of this archetype as described in the instructions above (e.g. Donchian channels, Supertrend, Bollinger-Keltner squeeze, Linear Regression slope, or HTF EMA).
Do NOT copy or default to generic swing sweep templates!
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
            # UPGRADE 5: Second parent for structured cross-pollination
            second_parent_section = ""
            if second_parent_code:
                second_parent_section = f"""
PARENT B (REGIME FILTER & EXIT MODEL PROVIDER):
```python
{second_parent_code[:1500]}
```
STRUCTURED SYNTHESIZER CONTRACT:
- Inherit ENTRY TRIGGER & LEVEL CALCULATIONS from PARENT A (Champion).
- Inherit REGIME FILTER, VOLATILITY GATING & EXIT MODEL (SL/TP runners) from PARENT B.
- Do NOT simply copy Parent A; create a true functional hybrid of both parents!
"""
            breeding_mandate = f"""
🧬 STRUCTURED GENETIC SYNTHESIS & CROSS-BREEDING MANDATE:
Below is PARENT A (Current Leaderboard Champion - proven positive Net R and high win rate):
```python
{champion_code}
```
{second_parent_section}
YOUR BREEDING MISSION:
Evolve a new high-alpha hybrid by cross-breeding Parent A with Archetype: "{archetype['name']}" ({archetype['concept']}){' and Parent B' if second_parent_code else ''}.
Guidelines for mutation:
1. Retain the core winning signal edge from Parent A (trigger structure, key price levels).
2. Inject regime filters and confirmation from {archetype['name']}{' and Parent B' if second_parent_code else ''} (e.g. session VWAP slope, ATR expansion, volume gating).
3. Evolve the risk management: Use ATR stop loss (min $4.00), scale with multi-target runners (tp1 at 1.8R-2.5R, tp2 at 3.5R-4.5R, or use_breakeven=True).
4. Eliminate noisy redundant conditions to ensure high trade execution fidelity (120 to 300 trades).
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
{purity_mandate}
{negative_section}
{failure_section}
{leaderboard_section}

MANDATORY INSTITUTIONAL RULES FOR PROFITABILITY:
1. STRICT ZERO LOOKAHEAD BIAS: No shift(-n), no center=True, no bfill(). All calculations causal.
   - ABSOLUTE BAN: NEVER use `center=True` in `.rolling()` (e.g. `df['high'].rolling(..., center=True)` peeks into future candles and is PERMANENTLY BLOCKED by the AST security parser).
   - Use strictly causal rolling: `df['high'].rolling(15).max().shift(1)` or call the built-in `find_swings(df)`.
2. REALISTIC CONFLUENCES (PREVENT ZERO-TRADE OVER-FILTERING):
   - Use 2 to 3 confluences MAXIMUM: [Macro Trend Context] + [Archetype Entry Trigger] + [Session Mask].
   - Examples of macro context: Higher-Timeframe Trend (`htf_trend_filter(df, '1h') >= 0` for longs, `<= 0` for shorts), Intraday VWAP (`df['close'] > vwap(df)`), or Daily Pivots (`levels['pp']`).
   - CRITICAL: Do NOT stack 5 or 6 simultaneous indicators with `&`. Over-filtering causes 0 trades to ever trigger!
   - Target 120 to 300 selective institutional trades over the 6-month train period (approx. 1 to 3 high-conviction trades per day). This ensures sufficient sample size and cumulative R-multiple to achieve high-yield alpha (>100R).
3. ANTI-BLEED / NON-REPEATING TRANSITION SIGNALS (PREVENT OVERTRADING CHURN):
   - Entry signals MUST trigger ONLY on the FIRST candle transition of a setup:
     `raw_bull = setup_condition & sess`
     `bull_signal = raw_bull & (~raw_bull.shift(1).fillna(False))`
   - NEVER fire repeatedly on consecutive bars during a rolling window!
4. INSTITUTIONAL STOP LOSS, BREAKEVEN & MULTI-TARGET RUNNERS:
    - ALL trade entries are taken strictly at the CANDLE CLOSE (`df['close']`).
    - Stop Loss MUST use a FULL ATR multiplier (1.2x to 2.2x) — NOT a tiny fractional multiplier:
      `sl_long = np.minimum(df['low'], swing_lows) - (1.5 * atr(df, 14))`
      `sl_short = np.maximum(df['high'], swing_highs) + (1.5 * atr(df, 14))`
    - The stop distance from entry MUST be at least $4.00 (Gold spreads + slippage = $0.25):
      `risk_long = np.maximum(df['close'] - sl_long, 4.00)`
      `risk_short = np.maximum(sl_short - df['close'], 4.00)`
    - Multi-Target Scaling & Breakeven Protection:
      # OPTIMAL GOLD PAYOUT (Empirical peak sweet spot is 3.8R to 4.4R):
      `df['tp1_long'] = df['close'] + (risk_long * 3.8)   # Optimal target (3.8R - 4.4R): Captures full Gold daily trend expansion`
      `df['tp1_short'] = df['close'] - (risk_short * 3.8)`
      # Optional multi-target runner (if desired):
      # `df['tp2_long'] = df['close'] + (risk_long * 4.4)`
      # `df['use_breakeven'] = True`
    - NEVER use buffer_factor < 1.0 for ATR multipliers! Values like 0.20, 0.25, 0.35 are FORBIDDEN.
    - NEVER invert Risk-to-Reward or use micro-stops (< $4.00)!
5. Mandatory Session Gating: Wrap all entries in `session_mask(df, 'london_ny')` (Continuous 06:00-18:00 UTC institutional window - NO midday blackout!). Or use `session_mask(df, 'london_fix')` for 15:00 UTC PM Fixation.
6. AVAILABLE ADVANCED INSTITUTIONAL INDICATORS:
   - `volume_profile_levels(df)`: Previous day's Volume Profile ('poc', 'vah', 'val' strictly shifted).
   - `htf_swings(df, swing_len=5, timeframe='15min')`: Causal 15-minute Higher-Timeframe Swings.
   - `session_compression(df, 'asia')`: Pre-London compression ratio (< 0.75 indicates coil).
   - `absorption_volume(df)`: Boolean Series indicating institutional stopping volume absorption at S1/R1.
   - `volatility_ratio(df, 5, 30)`: Dynamic ATR expansion ratio (> 1.10 = explosive expansion).
   - `htf_order_blocks(df, '1h')`: Returns (bullish_ob_level, bearish_ob_level) for Higher Timeframe.
   - `macro_dxy(df)`: Synthetic causal Dollar Index (DXY) proxy for cross-market divergence.
   - `macro_us10y(df)`: Synthetic causal US 10-Year Yield proxy.
   - `order_flow_imbalance(df, period=14)`: Tick-level order flow imbalance (Positive = Buying pressure).
   - `realized_volatility(df, period=20)`: Annualized rolling realized volatility.
   - `daily_levels(df)`: Daily Floor Pivots ('pdh', 'pdl', 'pdc', 'pivot', 'r1', 's1', 'r2', 's2').
   - `vwap(df)`: Intraday volume-weighted average price.
   - `macd(df['close'], 12, 26, 9)` or `(10, 20, 7)`.
   - `find_swings(df, swing_len=7)`.
6. Define calculate_signals(df) returning df with 'bull_signal', 'bear_signal', 'sl_long', 'sl_short', 'tp1_long', 'tp1_short', 'tp2_long', 'tp2_short', 'use_breakeven'.
7. MUST end with `return df`.
8. Return ONLY executable Python code in ```python ... ``` block.
"""
        # Call with dynamic temperature (0.7 default, up to 0.85 on mutation shifts)
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
                temperature=float(temperature)
            )
            raw_code = _clean_code_response(raw_resp)
            if raw_code and "def calculate_signals" not in raw_code:
                indented_body = "\n".join(f"    {line}" for line in raw_code.split("\n"))
                code = f"def calculate_signals(df):\n{indented_body}\n    return df\n"
            else:
                code = raw_code
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
        rr = round(random.choice([1.8, 2.0, 2.2, 2.5]), 2)
        runner_rr = round(rr * 2.5, 1)
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
            if 'tp2_long' not in mutated and 'tp1_long' in mutated:
                mutated = re.sub(
                    r"(df\['tp1_long'\]\s*=\s*df\['close'\]\s*\+\s*\(?(?:risk_long|risk_l)\s*\*\s*[\d\.]+\)?)",
                    rf"\g<1>\n    df['tp2_long'] = df['close'] + (risk_long * {runner_rr})\n    df['use_breakeven'] = True",
                    mutated
                )
                mutated = re.sub(
                    r"(df\['tp1_short'\]\s*=\s*df\['close'\]\s*-\s*\(?(?:risk_short|risk_s)\s*\*\s*[\d\.]+\)?)",
                    rf"\g<1>\n    df['tp2_short'] = df['close'] - (risk_short * {runner_rr})\n    df['use_breakeven'] = True",
                    mutated
                )
            if mutated != champion_code:
                return mutated

        idx = archetype_idx % len(load_archetypes())
        arch_id = load_archetypes()[idx].get('id', '')

        if idx == 0 or arch_id == 'sequential_smc_fvg_state_machine':
            return f'''def calculate_signals(df):
    sess = session_mask(df, 'london_ny')
    macro = htf_trend_filter(df, '1h')
    sw_highs, sw_lows = find_swings(df, swing_len={sw_len})
    bull_fvg_top, bull_fvg_bot, bear_fvg_top, bear_fvg_bot = find_fvgs(df)
    vwap_line = vwap(df)
    
    ssl_sweep = (df['low'] < sw_lows) & (df['close'] > sw_lows)
    bsl_sweep = (df['high'] > sw_highs) & (df['close'] < sw_highs)
    
    armed_long = ssl_sweep.rolling(8, min_periods=1).max() == 1
    armed_short = bsl_sweep.rolling(8, min_periods=1).max() == 1
    
    fvg_touch_long = (df['low'] <= bull_fvg_top) & (df['close'] > bull_fvg_bot)
    fvg_touch_short = (df['high'] >= bear_fvg_bot) & (df['close'] < bear_fvg_top)
    
    raw_bull = armed_long & fvg_touch_long & (df['close'] > vwap_line) & (macro >= 0) & sess
    raw_bear = armed_short & fvg_touch_short & (df['close'] < vwap_line) & (macro <= 0) & sess
    
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    atr_val = atr(df, {atr_period})
    sl_long = np.minimum(df['low'], sw_lows) - ({atr_mult} * atr_val)
    sl_short = np.maximum(df['high'], sw_highs) + ({atr_mult} * atr_val)
    
    risk_l = np.maximum(df['close'] - sl_long, {min_dist})
    risk_s = np.maximum(sl_short - df['close'], {min_dist})
    
    df['sl_long'] = df['close'] - risk_l
    df['sl_short'] = df['close'] + risk_s
    df['use_breakeven'] = True
    df['use_trailing'] = True
    df['tp1_long'] = df['close'] + (risk_l * {rr})
    df['tp2_long'] = df['close'] + (risk_l * {runner_rr})
    df['tp1_short'] = df['close'] - (risk_s * {rr})
    df['tp2_short'] = df['close'] - (risk_s * {runner_rr})
    return df
'''

        elif idx == 1 or arch_id == 'multi_timeframe_key_levels':
            return f'''def calculate_signals(df):
    sess = session_mask(df, 'london_ny')
    macro = htf_trend_filter(df, '1h')
    levels = daily_levels(df)
    vwap_line = vwap(df)
    atr_val = atr(df, {atr_period})
    
    s1_sweep = (df['low'] < levels['s1']) & (df['close'] > levels['s1'])
    r1_sweep = (df['high'] > levels['r1']) & (df['close'] < levels['r1'])
    
    raw_bull = s1_sweep & (df['close'] > vwap_line) & (macro >= 0) & sess
    raw_bear = r1_sweep & (df['close'] < vwap_line) & (macro <= 0) & sess
    
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    risk_l = np.maximum({atr_mult} * atr_val, {min_dist})
    risk_s = np.maximum({atr_mult} * atr_val, {min_dist})
    
    df['sl_long'] = df['close'] - risk_l
    df['sl_short'] = df['close'] + risk_s
    df['use_breakeven'] = True
    df['use_trailing'] = True
    df['tp1_long'] = df['close'] + (risk_l * {rr})
    df['tp2_long'] = df['close'] + (risk_l * {runner_rr})
    df['tp1_short'] = df['close'] - (risk_s * {rr})
    df['tp2_short'] = df['close'] - (risk_s * {runner_rr})
    return df
'''

        elif idx == 2 or arch_id == 'regime_gated_execution':
            return f'''def calculate_signals(df):
    sess = session_mask(df, 'london_ny')
    macro = htf_trend_filter(df, '1h')
    er = efficiency_ratio(df['close'], 20)
    adx_val, _, _ = adx(df, {atr_period})
    st_line, st_dir = supertrend(df, 10, 3.0)
    atr_val = atr(df, {atr_period})
    
    st_bull_cross = (st_dir == 1) & (st_dir.shift(1) <= 0)
    st_bear_cross = (st_dir == -1) & (st_dir.shift(1) >= 0)
    
    regime_ok = (er > 0.28) & (adx_val > {adx_thresh}) & sess
    
    raw_bull = st_bull_cross & regime_ok & (macro >= 0)
    raw_bear = st_bear_cross & regime_ok & (macro <= 0)
    
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    risk_l = np.maximum(df['close'] - st_line, {min_dist})
    risk_s = np.maximum(st_line - df['close'], {min_dist})
    risk_l = np.maximum(risk_l, {atr_mult} * atr_val)
    risk_s = np.maximum(risk_s, {atr_mult} * atr_val)
    
    df['sl_long'] = df['close'] - risk_l
    df['sl_short'] = df['close'] + risk_s
    df['use_breakeven'] = True
    df['use_trailing'] = True
    df['tp1_long'] = df['close'] + (risk_l * {rr})
    df['tp2_long'] = df['close'] + (risk_l * {runner_rr})
    df['tp1_short'] = df['close'] - (risk_s * {rr})
    df['tp2_short'] = df['close'] - (risk_s * {runner_rr})
    return df
'''

        elif arch_id == 'donchian_turtle_momentum' or idx == 8:
            return f'''def calculate_signals(df):
    sess = session_mask(df, 'london_ny')
    macro = htf_trend_filter(df, '1h')
    d_up, d_dn, d_mid = donchian_channels(df, period=20)
    atr_val = atr(df, {atr_period})
    er = efficiency_ratio(df['close'], 20)
    
    bull_break = (df['close'] > d_up.shift(1)) & (~(df['close'].shift(1) > d_up.shift(2)))
    bear_break = (df['close'] < d_dn.shift(1)) & (~(df['close'].shift(1) < d_dn.shift(2)))
    
    raw_bull = bull_break & (er > 0.25) & (macro >= 0) & sess
    raw_bear = bear_break & (er > 0.25) & (macro <= 0) & sess
    
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    risk_l = np.maximum(df['close'] - d_mid, {min_dist})
    risk_s = np.maximum(d_mid - df['close'], {min_dist})
    risk_l = np.maximum(risk_l, {atr_mult} * atr_val)
    risk_s = np.maximum(risk_s, {atr_mult} * atr_val)
    
    df['sl_long'] = df['close'] - risk_l
    df['sl_short'] = df['close'] + risk_s
    df['use_breakeven'] = True
    df['use_trailing'] = True
    df['tp1_long'] = df['close'] + (risk_l * {rr})
    df['tp2_long'] = df['close'] + (risk_l * {runner_rr})
    df['tp1_short'] = df['close'] - (risk_s * {rr})
    df['tp2_short'] = df['close'] - (risk_s * {runner_rr})
    return df
'''

        elif arch_id == 'stochastic_macd_momentum' or idx == 13:
            return f'''def calculate_signals(df):
    sess = session_mask(df, 'london_ny')
    macro = htf_trend_filter(df, '1h')
    k, d = stochastic(df, 14, 3)
    macd_line, macd_sig, macd_hist = macd(df['close'], 12, 26, 9)
    atr_val = atr(df, {atr_period})
    
    bull_cross = (k.shift(1) < 30) & (k > d) & (k.shift(1) <= d.shift(1)) & (macd_hist > 0)
    bear_cross = (k.shift(1) > 70) & (k < d) & (k.shift(1) >= d.shift(1)) & (macd_hist < 0)
    
    raw_bull = bull_cross & (macro >= 0) & sess
    raw_bear = bear_cross & (macro <= 0) & sess
    
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    risk_l = np.maximum({atr_mult} * atr_val, {min_dist})
    risk_s = np.maximum({atr_mult} * atr_val, {min_dist})
    
    df['sl_long'] = df['close'] - risk_l
    df['sl_short'] = df['close'] + risk_s
    df['use_breakeven'] = True
    df['use_trailing'] = True
    df['tp1_long'] = df['close'] + (risk_l * {rr})
    df['tp2_long'] = df['close'] + (risk_l * {runner_rr})
    df['tp1_short'] = df['close'] - (risk_s * {rr})
    df['tp2_short'] = df['close'] - (risk_s * {runner_rr})
    return df
'''

        elif arch_id == 'opening_range_breakout_orb' or idx == 15:
            return f'''def calculate_signals(df):
    sess = session_mask(df, 'london_ny')
    macro = htf_trend_filter(df, '1h')
    atr_val = atr(df, {atr_period})
    times = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('dt', df.index))
    mins = times.hour * 60 + times.minute
    
    # 15m opening windows
    is_orb_london = (7 * 60 <= mins) & (mins < 7 * 60 + 15)
    is_orb_ny = (12 * 60 + 30 <= mins) & (mins < 12 * 60 + 45)
    is_orb = is_orb_london | is_orb_ny
    
    orb_high = df['high'].where(is_orb).rolling(3, min_periods=1).max().ffill()
    orb_low = df['low'].where(is_orb).rolling(3, min_periods=1).min().ffill()
    
    bull_break = (df['close'] > orb_high) & (~(df['close'].shift(1) > orb_high.shift(1))) & (~is_orb)
    bear_break = (df['close'] < orb_low) & (~(df['close'].shift(1) < orb_low.shift(1))) & (~is_orb)
    vol_ok = rvol(df, 20) > 1.15
    
    raw_bull = bull_break & vol_ok & (macro >= 0) & sess
    raw_bear = bear_break & vol_ok & (macro <= 0) & sess
    
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    risk_l = np.maximum({atr_mult} * atr_val, {min_dist})
    risk_s = np.maximum({atr_mult} * atr_val, {min_dist})
    
    df['sl_long'] = df['close'] - risk_l
    df['sl_short'] = df['close'] + risk_s
    df['use_breakeven'] = True
    df['use_trailing'] = True
    df['tp1_long'] = df['close'] + (risk_l * {rr})
    df['tp2_long'] = df['close'] + (risk_l * {runner_rr})
    df['tp1_short'] = df['close'] - (risk_s * {rr})
    df['tp2_short'] = df['close'] - (risk_s * {runner_rr})
    return df
'''

        elif arch_id == 'dual_volatility_ratio_squeeze' or idx == 20:
            return f'''def calculate_signals(df):
    sess = session_mask(df, 'london_ny')
    macro = htf_trend_filter(df, '1h')
    vr = volatility_ratio(df, 5, 30)
    atr_val = atr(df, {atr_period})
    d_up, d_dn, d_mid = donchian_channels(df, 20)
    vwap_line = vwap(df)
    
    compressed = (vr < 0.70).rolling(8, min_periods=1).max() == 1
    expansion = compressed & (vr > 1.05) & (vr.shift(1) <= 1.05)
    
    raw_bull = expansion & (df['close'] >= d_up.shift(1)) & (df['close'] > vwap_line) & (macro >= 0) & sess
    raw_bear = expansion & (df['close'] <= d_dn.shift(1)) & (df['close'] < vwap_line) & (macro <= 0) & sess
    
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    risk_l = np.maximum(df['close'] - d_mid, {min_dist})
    risk_s = np.maximum(d_mid - df['close'], {min_dist})
    risk_l = np.maximum(risk_l, {atr_mult} * atr_val)
    risk_s = np.maximum(risk_s, {atr_mult} * atr_val)
    
    df['sl_long'] = df['close'] - risk_l
    df['sl_short'] = df['close'] + risk_s
    df['use_breakeven'] = True
    df['use_trailing'] = True
    df['tp1_long'] = df['close'] + (risk_l * {rr})
    df['tp2_long'] = df['close'] + (risk_l * {runner_rr})
    df['tp1_short'] = df['close'] - (risk_s * {rr})
    df['tp2_short'] = df['close'] - (risk_s * {runner_rr})
    return df
'''

        elif arch_id == 'vwap_2sigma_band_walking' or idx == 23:
            return f'''def calculate_signals(df):
    sess = session_mask(df, 'london_ny')
    macro = htf_trend_filter(df, '1h')
    vwap_line = vwap(df)
    z_val = zscore(df['close'] - vwap_line, 20)
    atr_val = atr(df, {atr_period})
    adx_val, _, _ = adx(df, 14)
    er = efficiency_ratio(df['close'], 20)
    
    bull_drive = (z_val >= 1.8) & (z_val.shift(1) < 1.8) & (adx_val > 22) & (er > 0.28)
    bear_drive = (z_val <= -1.8) & (z_val.shift(1) > -1.8) & (adx_val > 22) & (er > 0.28)
    
    raw_bull = bull_drive & (macro >= 0) & sess
    raw_bear = bear_drive & (macro <= 0) & sess
    
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    risk_l = np.maximum(df['close'] - vwap_line, {min_dist})
    risk_s = np.maximum(vwap_line - df['close'], {min_dist})
    risk_l = np.maximum(risk_l, {atr_mult} * atr_val)
    risk_s = np.maximum(risk_s, {atr_mult} * atr_val)
    
    df['sl_long'] = df['close'] - risk_l
    df['sl_short'] = df['close'] + risk_s
    df['use_breakeven'] = True
    df['use_trailing'] = True
    df['tp1_long'] = df['close'] + (risk_l * {rr})
    df['tp2_long'] = df['close'] + (risk_l * {runner_rr})
    df['tp1_short'] = df['close'] - (risk_s * {rr})
    df['tp2_short'] = df['close'] - (risk_s * {runner_rr})
    return df
'''

        else:
            return f'''def calculate_signals(df):
    sess = session_mask(df, 'london_ny')
    macro = htf_trend_filter(df, '1h')
    sw_highs, sw_lows = find_swings(df, swing_len={sw_len})
    atr_val = atr(df, {atr_period})
    ema_fast = ema(df['close'], {ema_fast})
    ema_slow = ema(df['close'], {ema_slow})
    vol_filter = rvol(df, 20) > {rvol_thresh}
    
    ssl_sweep = (df['low'] < sw_lows) & (df['close'] > sw_lows)
    bsl_sweep = (df['high'] > sw_highs) & (df['close'] < sw_highs)
    trend_bull = (ema_fast > ema_slow) & (macro >= 0)
    trend_bear = (ema_fast < ema_slow) & (macro <= 0)
    
    raw_bull = ssl_sweep & trend_bull & vol_filter & sess
    raw_bear = bsl_sweep & trend_bear & vol_filter & sess
    
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    risk_l = np.maximum(df['close'] - (sw_lows - ({atr_mult} * atr_val)), {min_dist})
    risk_s = np.maximum((sw_highs + ({atr_mult} * atr_val)) - df['close'], {min_dist})
    
    df['sl_long'] = df['close'] - risk_l
    df['sl_short'] = df['close'] + risk_s
    df['use_breakeven'] = True
    df['use_trailing'] = True
    df['tp1_long'] = df['close'] + (risk_l * {rr})
    df['tp2_long'] = df['close'] + (risk_l * {runner_rr})
    df['tp1_short'] = df['close'] - (risk_s * {rr})
    df['tp2_short'] = df['close'] - (risk_s * {runner_rr})
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
        """Injects institutional ATR SL/TP and multi-target scaling if missing from calculate_signals."""
        injection = """
    # Risk Officer Injected Institutional ATR Risk & Multi-Target Runner Protection (Min $4.00 Stop Distance)
    atr_risk = atr(df, period=14)
    if 'sl_long' not in df.columns:
        df['sl_long'] = np.where(df['bull_signal'], df['low'] - 1.5 * atr_risk, np.nan)
        risk_l = np.maximum(df['close'] - df['sl_long'], 4.00)
        df['tp1_long'] = np.where(df['bull_signal'], df['close'] + 2.0 * risk_l, np.nan)
        df['tp2_long'] = np.where(df['bull_signal'], df['close'] + 5.5 * risk_l, np.nan)
        df['use_breakeven'] = True
    elif 'tp2_long' not in df.columns and 'tp1_long' in df.columns:
        risk_l = np.maximum(df['close'] - df['sl_long'], 4.00)
        df['tp2_long'] = np.where(df['bull_signal'], df['close'] + 5.5 * risk_l, np.nan)
        df['use_breakeven'] = True

    if 'sl_short' not in df.columns:
        df['sl_short'] = np.where(df['bear_signal'], df['high'] + 1.5 * atr_risk, np.nan)
        risk_s = np.maximum(df['sl_short'] - df['close'], 4.00)
        df['tp1_short'] = np.where(df['bear_signal'], df['close'] - 2.0 * risk_s, np.nan)
        df['tp2_short'] = np.where(df['bear_signal'], df['close'] - 5.5 * risk_s, np.nan)
        df['use_breakeven'] = True
    elif 'tp2_short' not in df.columns and 'tp1_short' in df.columns:
        risk_s = np.maximum(df['sl_short'] - df['close'], 4.00)
        df['tp2_short'] = np.where(df['bear_signal'], df['close'] - 5.5 * risk_s, np.nan)
        df['use_breakeven'] = True
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
    Evaluates a 12-point grid of (atr_mult, rr_ratio) on real MetaTrader 5 ECN broker data
    to discover the mathematical alpha peak for any strategy candidate.
    """

    @staticmethod
    def _mutate_code_params(code: str, am: float, rr: float, lookback_param: Optional[Tuple[str, int]] = None) -> str:
        """Robustly mutates ATR stop buffers, lookbacks, and TP risk multipliers with multi-target runners."""
        mutated = code
        # 0. Lookback sensitivity mutation
        if lookback_param:
            p_name, p_val = lookback_param
            mutated = re.sub(rf'\b({p_name})\s*=\s*\d+', rf'\g<1> = {p_val}', mutated)

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
        runner_rr = max(4.5, round(rr * 2.2, 1))
        if 'tp2_long' in mutated or 'tp2_short' in mutated:
            mutated = re.sub(
                r"(df\['tp1_long'\]\s*=\s*df\['close'\]\s*\+\s*\(?risk_\w+\s*\*\s*)[\d\.]+",
                rf"\g<1>{rr}",
                mutated
            )
            mutated = re.sub(
                r"(df\['tp1_short'\]\s*=\s*df\['close'\]\s*-\s*\(?risk_\w+\s*\*\s*)[\d\.]+",
                rf"\g<1>{rr}",
                mutated
            )
            mutated = re.sub(
                r"(df\['tp2_long'\]\s*=\s*df\['close'\]\s*\+\s*\(?risk_\w+\s*\*\s*)[\d\.]+",
                rf"\g<1>{runner_rr}",
                mutated
            )
            mutated = re.sub(
                r"(df\['tp2_short'\]\s*=\s*df\['close'\]\s*-\s*\(?risk_\w+\s*\*\s*)[\d\.]+",
                rf"\g<1>{runner_rr}",
                mutated
            )
        else:
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
            if 'tp1_long' in mutated and 'tp2_long' not in mutated:
                mutated = re.sub(
                    r"(df\['tp1_long'\]\s*=\s*df\['close'\]\s*\+\s*\(?(?:risk_\w+|risk[ls]|sl_dist(?:_\w+)?)\s*\*\s*[\d\.]+\)?)",
                    rf"\g<1>\n    df['tp2_long'] = df['close'] + (risk_l * {runner_rr})\n    df['use_breakeven'] = True\n    df['use_trailing'] = True",
                    mutated
                )
                mutated = re.sub(
                    r"(df\['tp1_short'\]\s*=\s*df\['close'\]\s*-\s*\(?(?:risk_\w+|risk[ls]|sl_dist(?:_\w+)?)\s*\*\s*[\d\.]+\)?)",
                    rf"\g<1>\n    df['tp2_short'] = df['close'] - (risk_s * {runner_rr})\n    df['use_breakeven'] = True\n    df['use_trailing'] = True",
                    mutated
                )

        if 'use_trailing' not in mutated and 'use_breakeven' in mutated:
            mutated = mutated.replace("df['use_breakeven'] = True", "df['use_breakeven'] = True\n    df['use_trailing'] = True")

        return mutated

    @staticmethod
    def sweep_and_optimize(base_code: str, df: pd.DataFrame) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], Dict[str, float]]:
        """
        Two-Tier Fast Vectorized Parameter Optimizer:
        Stage 1: Quick lookback sensitivity sweep (swing_len in [5, 7, 10] or period in [10, 14, 20]).
        Stage 2: Asymmetrical runner grid sweep (3 ATRs x 5 RRs with trailing stops).
        """
        # Fast 1-Pass Baseline Prune:
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
        base_pf = min(float(base_stats.get('profit_factor', 1.0)), 5.0)
        base_dd_r = float(base_stats.get('max_drawdown', 0.0) / 1000.0)
        base_wr = float(base_stats.get('win_rate', 0.0))

        # High-Speed Prune: If baseline is severely bleeding (Net R < -15.0R) or has too few trades (< 8)
        if base_net_r < -15.0 or len(base_trades) < 8:
            return base_code, base_stats, base_trades, base_monthly

        best_score = (base_net_r * 2.5) + (base_months_10 * 10.0) - (base_dd_r * 2.0) + (base_pf * 8.0) + (base_wr * 0.2)
        if len(base_trades) < 30:
            best_score *= max(0.1, len(base_trades) / 30.0)

        # STAGE 1: Lookback Sensitivity Coordinate Sweep
        lookback_candidates = []
        if 'swing_len' in base_code:
            lookback_candidates = [('swing_len', 5), ('swing_len', 7), ('swing_len', 10)]
        elif 'period=' in base_code or 'period =' in base_code:
            lookback_candidates = [('period', 10), ('period', 14), ('period', 20)]

        for lb in lookback_candidates:
            mut_lb = ParameterGridSweeper._mutate_code_params(base_code, 1.5, 3.0, lookback_param=lb)
            if mut_lb != base_code:
                lb_res = execute_strategy(mut_lb, df)
                if lb_res.get('success'):
                    lb_tr = lb_res.get('trades', [])
                    if len(lb_tr) >= 10:
                        lb_st = lb_res.get('stats', {})
                        lb_m = compute_monthly_r_breakdown(lb_tr)
                        lb_net_r = sum(lb_m.values()) if lb_m else lb_st.get('total_pnl', 0.0) / 1000.0
                        if lb_net_r > base_net_r:
                            base_code = mut_lb
                            base_net_r = lb_net_r

        # STAGE 2: High-Speed Alpha Runner Grid (3 ATRs x 5 RRs)
        atr_mults = [1.4, 1.8, 2.0]
        rr_ratios = [2.5, 3.0, 3.5, 4.0, 4.5]

        consecutive_dead = 0
        for am in atr_mults:
            if consecutive_dead >= 5 and best_score <= 0:
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
                        if consecutive_dead >= 5 and best_score <= 0:
                            break
                        continue
                    else:
                        consecutive_dead = 0
                    if len(trades) >= 10:
                        monthly = compute_monthly_r_breakdown(trades)
                        months_10 = sum(1 for v in monthly.values() if v >= 10.0)
                        net_r = sum(monthly.values()) if monthly else stats.get('total_pnl', 0.0) / 1000.0
                        pf = min(float(stats.get('profit_factor', 1.0)), 5.0)
                        dd_r = float(stats.get('max_drawdown', 0.0) / 1000.0)
                        wr = float(stats.get('win_rate', 0.0))
                        
                        score = (net_r * 2.5) + (months_10 * 10.0) - (dd_r * 2.0) + (pf * 8.0) + (wr * 0.2)
                        if len(trades) < 30:
                            score *= max(0.1, len(trades) / 30.0)

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
                    f"  #{i+1} [{top.get('name', 'Champion')}]: {top.get('train_r', 0):+.1f}R (train) | "
                    f"PF: {top.get('train_pf', 0)} | WR: {top.get('train_win_rate', 0)}% | Trades: {top.get('train_trades', 0)}"
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

EMPIRICAL BACKTEST PERFORMANCE (100,000 Bar MT5 Broker ECN 5m Gold):
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
4. Mandatory Multi-Target Runners & Breakeven Protection:
   - MUST maintain or add `df['use_breakeven'] = True` (eliminates full losses after +1.2R move).
   - MUST define multi-target scaling:
     `df['tp1_long'] = df['close'] + (risk_long * 2.0)` (banks 50% profit)
     `df['tp2_long'] = df['close'] + (risk_long * 4.5)` (runner to capture mega-trends)
     `df['tp1_short'] = df['close'] - (risk_short * 2.0)`
     `df['tp2_short'] = df['close'] - (risk_short * 4.5)`
   - NEVER strip or collapse tp2 into a single low-multiple target!
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

