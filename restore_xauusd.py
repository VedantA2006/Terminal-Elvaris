import json

ml_archetypes = [
    {
        "name": "State-Space Regime Filter",
        "concept": "Using statistical regime switching to detect volatility shifts.",
        "instructions": "Implement a volatility regime filter based on statistical variance or GARCH-like proxies. Only trade when the regime indicates high momentum and low noise."
    },
    {
        "name": "Fractal Dimension Momentum",
        "concept": "Using Hurst exponent or fractal dimensions to gauge trend persistence.",
        "instructions": "Measure the fractal dimension of the price series. When the market is in a persistent trending state (Hurst > 0.5), execute breakout trades in the direction of the dominant trend."
    },
    {
        "name": "Spectral Density Breakout",
        "concept": "Detecting dominant cycle frequencies for mean reversion or breakout.",
        "instructions": "Use periodogram or spectral analysis proxies to identify the dominant cycle length. Enter trades when price deviates significantly from the dominant cyclical mean."
    },
    {
        "name": "Multivariate Gaussian Mixture Regime Filter",
        "concept": "Clustering market states into distinct Gaussian regimes.",
        "instructions": "Use multivariate proxies (volatility, momentum, volume) to define 'regimes'. Trade aggressively in the high-momentum regime and stay flat in the chop regime."
    },
    {
        "name": "Eigenvalue Covariance Trend Detector",
        "concept": "Detecting structural breaks using covariance matrix eigenvalues.",
        "instructions": "Use rolling covariance of returns and volume. A spike in the principal eigenvalue indicates a structural break and the start of a new institutional trend."
    }
]

with open("data/archetypes_XAUUSD.json", "w", encoding="utf-8") as f:
    json.dump(ml_archetypes, f, indent=4)

print("Restored ML concepts for XAUUSD!")
