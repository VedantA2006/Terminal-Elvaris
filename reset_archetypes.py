import json
import os

retail_ict_archetypes = [
    {
        "name": "Fair Value Gap (FVG) Trend Following",
        "concept": "Enter on pullbacks into 3-candle Fair Value Gaps in the direction of the macro trend.",
        "instructions": "Identify 3-candle FVGs (gap between candle 1 high and candle 3 low for longs). Filter by macro trend. Enter when price mitigates the FVG."
    },
    {
        "name": "Order Block (OB) Return",
        "concept": "Buy at bullish order blocks and sell at bearish order blocks aligned with the 1h trend.",
        "instructions": "Find the last down-candle before a strong impulsive up-move (Bullish OB). Wait for price to return and tap the OB during London/NY session."
    },
    {
        "name": "London Killzone Liquidity Sweep",
        "concept": "Sweep of Asian session high/low during the London Killzone followed by a reversal.",
        "instructions": "Track the Asian session High/Low (00:00 - 06:00 UTC). Enter a reversal trade if price sweeps the high/low during London Killzone (06:00 - 10:00 UTC)."
    },
    {
        "name": "VWAP Pullback Trend Following",
        "concept": "Pullbacks to the daily VWAP in a strongly trending market.",
        "instructions": "Calculate intraday VWAP. If price is firmly above VWAP and pulls back to touch it, enter long. Inverse for short."
    },
    {
        "name": "Volume Profile POC Rejection",
        "concept": "Mean reversion from the previous day's Point of Control (POC).",
        "instructions": "Use `volume_profile_levels`. Enter mean-reversion trades when price over-extends and rejects off the previous day's POC."
    },
    {
        "name": "Daily Pivot Point Bounce",
        "concept": "Reversals at standard Daily S1 and R1 levels.",
        "instructions": "Calculate Daily Floor Pivots. Look for exhaustion candle patterns at S1 for longs and R1 for shorts."
    },
    {
        "name": "Higher Timeframe (HTF) Swing Structure",
        "concept": "Aligning 5m entries with 15m or 1h higher timeframe swing highs and lows.",
        "instructions": "Use `htf_swings(df, swing_len=5, timeframe='15min')`. Only take longs when the HTF swing structure is making higher highs and higher lows."
    },
    {
        "name": "Asian Range Breakout",
        "concept": "Momentum breakout of the Asian session consolidation range.",
        "instructions": "Calculate Asian Range (00:00 - 06:00 UTC). Go long if price strongly breaks above the Asian High with volume expansion."
    },
    {
        "name": "MACD Momentum Crossover",
        "concept": "Standard MACD crossover aligned with a higher timeframe trend filter.",
        "instructions": "Use MACD (12, 26, 9). Enter on the crossover if it aligns with the macro trend and session volume is active."
    },
    {
        "name": "RSI Divergence",
        "concept": "RSI making higher lows while price makes lower lows.",
        "instructions": "Identify classic RSI divergence. Combine with a structural break of market structure (BMS) for entry."
    },
    {
        "name": "Inside Bar Breakout",
        "concept": "Volatility expansion after a period of inside bar compression.",
        "instructions": "Identify inside bars (current high < prev high and current low > prev low). Enter on the breakout of the inside bar."
    },
    {
        "name": "Opening Range Breakout (ORB)",
        "concept": "Breakout of the first 30-minute range of the NY Session.",
        "instructions": "Identify the high and low of the NY Open (13:30 - 14:00 UTC). Trade the breakout of this range."
    },
    {
        "name": "Stochastic Overbought/Oversold Reversal",
        "concept": "Reversals from deep stochastic extremes combined with support/resistance.",
        "instructions": "Use Stochastic Oscillator. Enter longs when Stochastic crosses up from below 20, provided it's at a key support level."
    },
    {
        "name": "Bollinger Band Mean Reversion",
        "concept": "Fading moves that pierce the outer Bollinger Bands.",
        "instructions": "Use Bollinger Bands (20, 2). Enter short when price closes outside the upper band and then closes back inside."
    },
    {
        "name": "ATR Volatility Breakout",
        "concept": "Explosive breakouts defined by dynamic ATR expansion.",
        "instructions": "Use `volatility_ratio(df, 5, 30)`. Enter momentum breakouts when the volatility ratio spikes > 1.20."
    }
]

instruments = ["US100", "SPX500", "EURUSD", "GBPUSD", "XAUUSD"]

# Overwrite fallback
os.makedirs("data", exist_ok=True)
with open("data/archetypes.json", "w", encoding="utf-8") as f:
    json.dump(retail_ict_archetypes, f, indent=4)

# Overwrite per-instrument
for inst in instruments:
    with open(f"data/archetypes_{inst}.json", "w", encoding="utf-8") as f:
        json.dump(retail_ict_archetypes, f, indent=4)

print("Successfully injected Retail ICT archetypes into all instruments!")
