import json

new_archetypes = [
    {
        "id": "xau_zscore_mean_reversion",
        "name": "XAU Z-Score Mean Reversion",
        "concept": "Identifies statistical over-extension using a rolling Z-Score and fades the extreme moves.",
        "instructions": "Calculate a rolling 20-period VWAP or EMA and its standard deviation. Compute the Z-Score. When Z-Score > 2.5, go short. When Z-Score < -2.5, go long. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_hurst_exponent_momentum",
        "name": "XAU Hurst Exponent Momentum",
        "concept": "Measures the long-term memory of the time series to identify structural trending regimes.",
        "instructions": "Approximate the Hurst Exponent (H) using rolling log returns. If H > 0.6, the market is trending. Trade momentum breakouts in the direction of the trend. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_kalman_filter_smoothing",
        "name": "XAU Kalman Filter Smoothing",
        "concept": "Applies a basic Kalman filter proxy using dynamic EMAs to eliminate noise.",
        "instructions": "Create a dynamically weighted moving average where alpha adjusts based on recent volatility. Trade crossovers of this smoothed line against price. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_volatility_regime_switch",
        "name": "XAU Volatility Regime Switch",
        "concept": "Switches between mean-reverting and trend-following logic based on the ATR/Standard Deviation ratio.",
        "instructions": "Measure the rolling 14-period ATR divided by the 14-period Standard Deviation. If ratio is high, use mean-reversion. If low, use trend-following. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_statistical_arbitrage_proxy",
        "name": "XAU Statistical Arbitrage Proxy",
        "concept": "Looks for short-term statistical dislocations between price and a synthetic basket of moving averages.",
        "instructions": "Calculate a basket of 3 different moving averages (SMA, EMA, WMA). Find the z-score of the price relative to the basket mean. Fade extremes. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_fourier_cycle_analysis",
        "name": "XAU Fourier Cycle Analysis",
        "concept": "Uses dominant cycle lengths derived from price oscillation to time entries.",
        "instructions": "Calculate the dominant cycle length using a stochastic oscillator or MACD zero-cross frequency. Trade in the direction of the dominant cycle. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_pca_momentum",
        "name": "XAU Principal Component Momentum",
        "concept": "Combines multiple momentum indicators (RSI, MACD, ROC) into a single composite score.",
        "instructions": "Normalize RSI, MACD Histogram, and ROC. Sum them to create a composite momentum score. Trade breakouts when the composite score crosses a threshold. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_volume_weighted_macd",
        "name": "XAU Volume Weighted MACD",
        "concept": "A MACD calculation where the underlying moving averages are Volume Weighted.",
        "instructions": "Calculate two VWAPs of different lengths. Take the difference to form a VW-MACD. Trade the crossovers. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_bid_ask_absorption_proxy",
        "name": "XAU Bid-Ask Absorption Proxy",
        "concept": "Detects institutional absorption by comparing price movement to volume delta.",
        "instructions": "If volume is extremely high but price makes a small range (doji), assume absorption. Trade in the opposite direction of the preceding micro-trend. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_fractal_dimension_index",
        "name": "XAU Fractal Dimension Index",
        "concept": "Measures market chop vs trend using fractal dimensions.",
        "instructions": "Calculate the ratio of the path length to the net distance traveled over N periods. Trade breakouts only when the fractal dimension drops, indicating a trend. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_adaptive_bollinger_bands",
        "name": "XAU Adaptive Bollinger Bands",
        "concept": "Bollinger Bands where the standard deviation multiplier scales dynamically with market volatility.",
        "instructions": "Increase the BB multiplier when ATR is high, decrease it when ATR is low. Trade mean reversion off the bands. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_price_volume_divergence",
        "name": "XAU Price-Volume Divergence",
        "concept": "Searches for structural divergence between price highs and volume peaks.",
        "instructions": "Identify higher highs in price but lower highs in volume over a 20-period rolling window. Short the divergence. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_order_flow_imbalance",
        "name": "XAU Order Flow Imbalance",
        "concept": "Tracks sequential up/down ticks to approximate order flow aggression.",
        "instructions": "Count consecutive higher closes vs lower closes. When a sequence reaches extreme imbalance (e.g., 5 in a row with increasing volume), trade the exhaustion fade. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_markov_chain_regime",
        "name": "XAU Markov Chain Regime",
        "concept": "Models probability of up vs down bars using recent transition matrices.",
        "instructions": "Calculate the 20-period transition probability of Up->Up vs Down->Up. Trade only when the probability matrix strongly favors one direction. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    },
    {
        "id": "xau_synthetic_options_pricing",
        "name": "XAU Synthetic Options Pricing",
        "concept": "Uses Black-Scholes inspired historical volatility to map expected standard deviations.",
        "instructions": "Calculate 20-period annualized historical volatility. Map 1-sigma bounds on the 5-minute chart. Trade mean reversion at the bounds. MUST strictly avoid trading during high-impact news events and weekend market closes. Ensure all lookback periods and thresholds are entirely unique to avoid duplicate strategies."
    }
]

with open('data/archetypes_XAUUSD.json', 'w', encoding='utf-8') as f:
    json.dump(new_archetypes, f, indent=4)

print("Successfully replaced all retail ICT archetypes with advanced quantitative archetypes!")
