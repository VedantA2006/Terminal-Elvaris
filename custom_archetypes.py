import json
import os

archetypes = {
    "US100": [
        {
            "name": "Opening Range Breakout (ORB) Momentum",
            "concept": "Tech-heavy momentum breakouts at the NY Open.",
            "instructions": "Calculate the high/low of the first 30 minutes of the NY session (13:30 - 14:00 UTC). Enter trades in the direction of the breakout with strong volume expansion."
        },
        {
            "name": "Tech Gap & Go Continuation",
            "concept": "Trading the continuation of overnight gaps.",
            "instructions": "Detect overnight gaps. If the price gaps up and holds the gap during the first 15 mins of NY open, buy the pullback to the VWAP."
        },
        {
            "name": "High-Beta VWAP Trend Following",
            "concept": "Aggressive trend following using VWAP and EMAs.",
            "instructions": "In strong trend days (price > 1h VWAP), buy the first intraday pullback to the 5m VWAP if accompanied by a MACD bullish cross."
        },
        {
            "name": "Volatility Contraction (VCP) Breakout",
            "concept": "Breakouts from tight consolidation zones.",
            "instructions": "Use Bollinger Bands bandwidth or ATR to detect extreme compression. Trade the explosive breakout when bandwidth expands."
        },
        {
            "name": "Late-Day Squeeze",
            "concept": "Momentum squeeze in the final hour of the NY session.",
            "instructions": "Identify tight ranges between 18:00 - 19:30 UTC. Trade the breakout into the 20:00 UTC close."
        }
    ],
    "SPX500": [
        {
            "name": "Overnight Gap Fade",
            "concept": "Mean reversion of overnight gaps.",
            "instructions": "Identify gaps between yesterday's close and today's NY open. If the gap is overextended, trade the fade back to yesterday's close."
        },
        {
            "name": "Statistical Mean Reversion",
            "concept": "Fading extreme moves using Bollinger Bands.",
            "instructions": "SPX is highly mean-reverting intraday. Short when price pierces the upper Bollinger Band (20, 2) and RSI > 75. Long when piercing lower band."
        },
        {
            "name": "Macro Session Trend Continuation",
            "concept": "Trading the dominant trend established in the morning.",
            "instructions": "Determine the trend direction from 13:30 - 15:30 UTC. Enter pullbacks in the direction of this trend during the afternoon session."
        },
        {
            "name": "Institutional VWAP Rejection",
            "concept": "Fading prices when they deviate too far from the VWAP.",
            "instructions": "Calculate the VWAP standard deviation bands. Mean-revert when price hits the 3rd standard deviation band."
        },
        {
            "name": "Daily Floor Pivot Bounce",
            "concept": "Reversals at key structural liquidity levels.",
            "instructions": "Use standard Daily Pivots. Buy exhaustion candles at S1/S2, short at R1/R2."
        }
    ],
    "EURUSD": [
        {
            "name": "London/NY Overlap Momentum",
            "concept": "High liquidity breakouts during the overlap session.",
            "instructions": "Trade breakouts of the London morning range specifically during the overlap window (13:00 - 16:00 UTC) when US volume enters."
        },
        {
            "name": "Frankfurt Fakeout (Stop Hunt)",
            "concept": "Trading the false breakout of the Asian range.",
            "instructions": "Track the Asian session range. If Frankfurt/early London breaks the high then immediately reverses back inside, short the false breakout."
        },
        {
            "name": "Interest Rate Divergence Proxy",
            "concept": "Using macro bond proxies (if available) or structural higher-timeframe momentum.",
            "instructions": "Align trades with the 1h and 4h trend (HTF Swings). Only take 5m entries that agree with this macro direction."
        },
        {
            "name": "RSI Divergence in Value Zones",
            "concept": "Mean reversion in a ranging environment.",
            "instructions": "When EURUSD is ranging (ADX < 25), use RSI divergence at previous day's high/low to enter mean-reversion trades."
        },
        {
            "name": "Order Block Mitigation",
            "concept": "Institutional order flow pullback trading.",
            "instructions": "Identify strong impulsive moves that leave behind a 15m order block. Enter on the first retest of this block."
        }
    ],
    "GBPUSD": [
        {
            "name": "Cable London Open Burst",
            "concept": "Aggressive momentum at the London open.",
            "instructions": "GBPUSD (Cable) often trends hard right at 07:00 UTC. Trade the breakout of the pre-London consolidation."
        },
        {
            "name": "High-Beta Fibonacci Pullback",
            "concept": "Trend continuation on deep pullbacks.",
            "instructions": "Identify a strong London morning trend. Wait for a deep 61.8% pullback and enter in the direction of the morning trend."
        },
        {
            "name": "Sterling Volatility Expansion",
            "concept": "ATR-based momentum trading.",
            "instructions": "Use a rolling volatility ratio (short ATR / long ATR). When the ratio spikes above 1.5, enter in the direction of the immediate 5m trend."
        },
        {
            "name": "Session High/Low Sweep",
            "concept": "Liquidity grabs at major session extremes.",
            "instructions": "Identify the high/low of the previous day. If price sweeps this level and sharply rejects, trade the reversal."
        },
        {
            "name": "MACD Zero-Line Cross",
            "concept": "Momentum shifts aligned with the 1h trend.",
            "instructions": "Filter for the 1h trend direction. Enter when the 5m MACD crosses the zero line in the direction of the 1h trend."
        }
    ]
}

os.makedirs("data", exist_ok=True)
for inst, archs in archetypes.items():
    with open(f"data/archetypes_{inst}.json", "w", encoding="utf-8") as f:
        json.dump(archs, f, indent=4)

print("Custom instrument-specific archetypes generated!")
