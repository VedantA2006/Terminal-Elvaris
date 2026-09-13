"""
TradingView Replica Web Terminal — Flask Application.

Endpoints:
  - /                         : Main TradingView duplicate UI
  - /api/ohlc                 : MetaTrader 5 ECN XAUUSD 5m candlestick data (100,000 bars)
  - /api/indicators           : Baseline indicator lines
  - /api/signals              : Baseline buy/sell markers
  - /api/trades               : Baseline trade records
  - /api/equity               : Baseline equity curve
  - /api/stats                : Baseline backtest stats
  - /api/strategy/default     : Default Python strategy template (Elvaris v2)
  - /api/strategy/run [POST]  : Live Python strategy runner + AST lookahead guard
  - /api/ai/generate [POST]   : AI Strategy Generator with hardcoded anti-lookahead rules
  - /api/ai/optimize_step [POST]: Autonomous Continuous Optimizer loop
"""

import os
import json
import math
from datetime import datetime
from flask import Flask, render_template, jsonify, request

from dotenv import load_dotenv
load_dotenv()

from download_data import load_or_download, load_timeframe_data, TIMEFRAME_MAP
from data_split import split_data
from strategy import calculate_signals
from backtest import run_backtest
from strategy_executor import execute_strategy
from ai_generator import generate_strategy_code, optimize_strategy_step, get_engine_telemetry
from default_strategy import DEFAULT_STRATEGY_CODE
from monte_carlo import run_monte_carlo
from leaderboard import load_leaderboard, get_strategy_by_id, add_strategy_to_leaderboard, upgrade_leaderboard_to_mt5_data
from portfolio_engine import get_portfolio_ensemble_data, build_portfolio_ensemble

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ---------------------------------------------------------------------------
# Global Cache
# ---------------------------------------------------------------------------
_cache = {}


def _initialise():
    """Run data loading and baseline calculations."""
    if 'raw_df' in _cache and _cache['raw_df'] is not None:
        return

    print("\n" + "=" * 65)
    print("  TRADINGVIEW PYTHON STRATEGY TERMINAL (MetaTrader 5 ECN XAUUSD)")
    print("=" * 65)

    # 1. Load raw MT5 broker data
    print("\n[1/3] Loading MT5 Broker XAUUSD 5m data ...")
    raw_df = load_or_download()
    _cache['raw_df'] = raw_df
    df_2026 = raw_df[raw_df.index >= '2026-01-01'].copy()
    _cache['df_2026'] = df_2026
    print(f"      {len(raw_df):,} real candles ready (2026 YTD: {len(df_2026):,} candles)")

    # 2. Calculate baseline signals (Elvaris v2)
    print("[2/3] Computing baseline Elvaris v2 indicators & signals ...")
    df = calculate_signals(raw_df)
    _cache['df'] = df

    # 3. Baseline backtest
    print("[3/3] Running baseline backtest ...")
    trades, equity_curve, stats = run_backtest(df, initial_capital=100000.0, lot_size=100.0, partial_tp=False)
    _cache['trades'] = trades
    _cache['equity'] = equity_curve
    _cache['stats'] = stats

    # 4. Compute chronological train/validation/test split
    print("[4/4] Computing train/validation/test split ...")
    train_df, val_df, test_df = split_data(raw_df)
    _cache['train_df'] = train_df
    _cache['val_df'] = val_df
    _cache['test_df'] = test_df

    # 5. Upgrade all leaderboard entries to MT5 broker backtest metrics (2026-only)
    try:
        upgrade_leaderboard_to_mt5_data(df_2026, train_df, val_df, test_df)
    except Exception as e:
        print(f"      [Leaderboard Upgrade Notice]: {e}")

    print(f"      Total Trades: {stats['total_trades']}, Win Rate: {stats['win_rate']}%, Net PnL: ${stats['total_pnl']:,.2f}")
    print("\n" + "=" * 65)
    print("  Terminal is LIVE: http://127.0.0.1:5000")
    print("=" * 65 + "\n")


def _clean(val):
    if val is None:
        return None
    if isinstance(val, (float, int)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 2)
    return val


# ---------------------------------------------------------------------------
# Core Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    _initialise()
    return render_template('index.html')


@app.route('/api/ohlc')
def api_ohlc():
    """Return OHLC candlestick data for Lightweight Charts. Supports ?timeframe= (e.g. 1m, 5m, 15m, 1h, 4h, 1d)."""
    _initialise()
    tf = request.args.get('timeframe', '5m').lower().strip()
    if tf and tf != '5m':
        try:
            df = load_timeframe_data(tf)
        except Exception as e:
            print(f"Warning loading timeframe {tf}: {e}")
            df = _cache['raw_df']
    else:
        df = _cache['raw_df']

    data = []
    for ts, row in df.iterrows():
        t = int(ts.timestamp())
        item = {
            'time': t,
            'open': _clean(row['open']),
            'high': _clean(row['high']),
            'low': _clean(row['low']),
            'close': _clean(row['close']),
            'volume': _clean(row.get('volume', 0)),
        }
        if 'spread' in row:
            item['spread'] = _clean(row['spread'])
        data.append(item)
    return jsonify(data)


@app.route('/api/timeframes')
def api_timeframes():
    """Return summary of all downloaded MT5 timeframes and their bar counts / ranges."""
    _initialise()
    res = {}
    from pathlib import Path
    data_dir = Path(__file__).parent / 'data'
    for tf_key, (mt5_tf, fname, _) in TIMEFRAME_MAP.items():
        fpath = data_dir / fname
        if fpath.exists():
            try:
                import pandas as pd
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                res[tf_key] = {
                    'timeframe': tf_key,
                    'mt5_code': mt5_tf,
                    'filename': fname,
                    'bars': len(df),
                    'start': str(df.index[0]),
                    'end': str(df.index[-1]),
                }
            except Exception:
                pass
    return jsonify(res)


@app.route('/api/indicators')
def api_indicators():
    """Return baseline indicator overlays (empty for clean chart)."""
    return jsonify({})


@app.route('/api/signals')
def api_signals():
    """Return markers strictly for trades taken (clean chart, no untaken signals)."""
    _initialise()
    import pandas as pd
    trades = _cache.get('trades', [])

    markers = []
    for t_data in trades:
        try:
            en_ts = int(pd.to_datetime(t_data['entry_time']).timestamp())
            is_long = t_data['direction'] == 'long'
            markers.append({
                'time': en_ts,
                'position': 'belowBar' if is_long else 'aboveBar',
                'color': '#089981' if is_long else '#f23645',
                'shape': 'arrowUp' if is_long else 'arrowDown',
                'text': f"#{t_data['id']} {'BUY' if is_long else 'SELL'}",
            })
            if t_data.get('exit_time') and t_data.get('exit_reason'):
                ex_ts = int(pd.to_datetime(t_data['exit_time']).timestamp())
                win = t_data['pnl'] > 0
                markers.append({
                    'time': ex_ts,
                    'position': 'aboveBar' if is_long else 'belowBar',
                    'color': '#089981' if win else '#f23645',
                    'shape': 'circle',
                    'text': f"{t_data['exit_reason']} ${t_data['pnl']:+,.0f}",
                })
        except Exception:
            pass

    markers.sort(key=lambda x: x['time'])
    return jsonify(markers)



@app.route('/api/trades')
def api_trades():
    _initialise()
    return jsonify(_cache['trades'])


@app.route('/api/equity')
def api_equity():
    _initialise()
    return jsonify(_cache['equity'])


@app.route('/api/stats')
def api_stats():
    _initialise()
    return jsonify(_cache['stats'])


@app.route('/api/strategy/default')
def api_strategy_default():
    """Returns the default Python strategy code."""
    return jsonify({'code': DEFAULT_STRATEGY_CODE})


@app.route('/api/strategy/run', methods=['POST'])
def api_strategy_run():
    """
    Executes user Python strategy code:
    1. AST Lookahead Bias Check
    2. Dynamic sandboxed calculation
    3. Bar-by-bar backtest
    """
    _initialise()
    data = request.get_json() or {}
    code = data.get('code', '')
    capital = float(data.get('capital', 100000.0))
    lot_size = float(data.get('lot_size', 100.0))
    spread = float(data.get('spread', 0.20))
    slippage = float(data.get('slippage', 0.05))

    if not code.strip():
        return jsonify({'success': False, 'message': 'Strategy code cannot be empty.'}), 400

    result = execute_strategy(
        code,
        _cache['raw_df'],
        initial_capital=capital,
        lot_size=lot_size,
        spread=spread,
        slippage=slippage
    )
    return jsonify(result)


@app.route('/api/ai/generate', methods=['POST'])
def api_ai_generate():
    """Generate a strategy using LLM with hardcoded anti-lookahead rules."""
    data = request.get_json() or {}
    provider = data.get('provider', 'omniroute')
    api_key = data.get('api_key', '')
    model = data.get('model', '')
    prompt = data.get('prompt', '')
    endpoint = data.get('endpoint', '')

    if not api_key and provider != 'omniroute':
        return jsonify({'success': False, 'message': 'API Key is required.'}), 400
    if not prompt:
        return jsonify({'success': False, 'message': 'Prompt is required.'}), 400

    res = generate_strategy_code(provider, api_key, model, prompt, endpoint_url=endpoint)
    return jsonify(res)


@app.route('/api/ai/optimize_step', methods=['POST'])
def api_ai_optimize_step():
    """Perform one step of autonomous strategy optimization."""
    _initialise()
    data = request.get_json() or {}
    provider = data.get('provider', 'omniroute')
    api_key = data.get('api_key', '')
    model = data.get('model', '')
    current_code = data.get('current_code', '')
    previous_stats = data.get('previous_stats', {})
    iteration = int(data.get('iteration', 1))
    endpoint = data.get('endpoint', '')

    if not api_key and provider != 'omniroute':
        return jsonify({'success': False, 'message': 'API Key is required.'}), 400
    if not current_code:
        return jsonify({'success': False, 'message': 'Current strategy code is required.'}), 400

    target_df = _cache.get('df_2026', _cache['raw_df'])
    res = optimize_strategy_step(
        provider, api_key, model, current_code, previous_stats, iteration, target_df, endpoint_url=endpoint
    )
    return jsonify(res)


@app.route('/api/leaderboard', methods=['GET'])
def api_leaderboard():
    """Returns ranked strategies list with monthly R and Monte Carlo metrics (2026 only)."""
    _initialise()
    board = load_leaderboard(_cache.get('df_2026'))
    return jsonify({'success': True, 'leaderboard': board})


@app.route('/api/leaderboard/load', methods=['POST'])
def api_leaderboard_load():
    """Loads a strategy from leaderboard by ID or index and executes it."""
    _initialise()
    data = request.get_json() or {}
    strat_id = str(data.get('id', '')).strip()
    idx = data.get('index')

    strat = None
    if strat_id and not strat_id.isdigit():
        strat = get_strategy_by_id(strat_id)

    if not strat and (idx is not None or strat_id.isdigit()):
        try:
            target_idx = int(idx if idx is not None else strat_id)
            lb = load_leaderboard(_cache.get('df_2026'))
            if 0 <= target_idx < len(lb):
                strat = lb[target_idx]
        except Exception:
            pass

    if not strat and strat_id:
        strat = get_strategy_by_id(strat_id)

    if not strat:
        return jsonify({'success': False, 'message': 'Strategy not found'}), 404

    code = strat.get('code')
    if not code or not code.strip():
        code = DEFAULT_STRATEGY_CODE

    target_df = _cache.get('df_2026')
    if target_df is None or len(target_df) < 100:
        target_df = _cache['raw_df']
    exec_res = execute_strategy(code, target_df)
    return jsonify({
        'success': True,
        'strategy': strat,
        'code': code,
        'exec_result': exec_res
    })


@app.route('/api/monte_carlo', methods=['POST'])
def api_monte_carlo():
    """Runs on-demand Monte Carlo bootstrap simulation on current or provided trades."""
    data = request.get_json() or {}
    trades = data.get('trades', [])
    if not trades:
        _initialise()
        trades = _cache.get('trades', [])
    simulations = int(data.get('simulations', 1000))
    mc_results = run_monte_carlo(trades, num_simulations=simulations)
    return jsonify({'success': True, 'monte_carlo': mc_results})


@app.route('/api/ai/start_generation', methods=['POST'])
def api_ai_start_generation():
    """
    Autonomous strategy creation step:
    1. Generates candidate code with Anti-Lookahead System Prompt
    2. Runs AST & Runtime lookahead verification
    3. Executes backtest against MT5 Broker ECN 5m Gold data (100,000 bars)
    4. Runs 1,000-path Monte Carlo stress test
    5. Evaluates monthly R-returns and saves candidate to Leaderboard
    """
    _initialise()
    data = request.get_json() or {}
    provider = data.get('provider', 'omniroute')
    api_key = data.get('api_key', '')
    model = data.get('model', '')
    prompt = data.get('prompt', '')
    endpoint = data.get('endpoint', '')

    if provider == 'omniroute':
        if not api_key:
            api_key = os.environ.get('OMNIROUTE_API_KEY', '')
            if not api_key:
                return jsonify({'success': False, 'message': 'OMNIROUTE_API_KEY env var is not set. See .env.example.'}), 400
        if not model or model in ('auto/best-coding', 'auto/best-reasoning', 'auto'):
            model = 'groq/qwen/qwen3.8-27b'
        if not endpoint:
            endpoint = 'http://localhost:20128/v1'
    elif not api_key:
        return jsonify({'success': False, 'message': 'API Key is required.'}), 400
    if not prompt:
        prompt = "Create a robust institutional quantitative strategy on 5m Gold using SMC sweeps, Daily Pivots, and Volatility Regimes."

    gen_res = generate_strategy_code(provider, api_key, model, prompt, endpoint_url=endpoint)
    if not gen_res.get('success'):
        return jsonify(gen_res), 400

    code = gen_res['code']

    # Execute backtest on 2026 data
    target_df = _cache.get('df_2026', _cache['raw_df'])
    exec_res = execute_strategy(code, target_df)
    if not exec_res.get('success'):
        return jsonify({
            'success': False,
            'error_type': exec_res.get('error_type', 'EXECUTION_ERROR'),
            'message': exec_res.get('message', 'Backtest execution failed'),
            'code': code
        }), 400

    trades = exec_res.get('trades', [])
    stats = exec_res.get('stats', {})

    if len(trades) < 5:
        return jsonify({
            'success': False,
            'message': f'Strategy conditions were too restrictive ({len(trades)} trades generated). Retrying with active liquidity sweep triggers...',
            'trades_count': len(trades)
        }), 400

    # Extract strategy concept title from first lines or prompt
    name_match = None
    lines = code.splitlines()
    for l in lines[:10]:
        cleaned_l = l.strip('# =*-\t\r\n')
        if (len(cleaned_l) > 4 
            and not cleaned_l.startswith('!') 
            and not cleaned_l.lower().startswith('import')
            and not cleaned_l.lower().startswith('from ')
            and not cleaned_l.lower().startswith('def ')):
            name_match = cleaned_l
            break
    strat_name = name_match if name_match else f"AI Quantitative Strategy #{int(datetime.utcnow().timestamp()) % 10000}"
    strat_concept = prompt[:80] + ('...' if len(prompt) > 80 else '')

    # Save to Leaderboard
    leaderboard_entry = add_strategy_to_leaderboard(
        name=strat_name,
        concept=strat_concept,
        code=code,
        stats=stats,
        trades=trades,
        author='Autonomous AI Generator',
        data_split='mt5_ecn_2026'
    )

    return jsonify({
        'success': True,
        'strategy': leaderboard_entry,
        'code': code,
        'exec_result': exec_res,
    })


# ---------------------------------------------------------------------------
# Multi-Agent Quant Lab Routes
# ---------------------------------------------------------------------------
from autonomous_research_loop import research_manager

@app.route('/api/research/start', methods=['POST'])
def api_research_start():
    data = request.get_json(silent=True) or {}
    rounds = int(data.get('rounds', 1000))
    provider = data.get('provider', 'omniroute')
    api_key = data.get('api_key', '')
    model = data.get('model', '')
    endpoint = data.get('endpoint', '')
    force_restart = bool(data.get('force_restart', False))

    omniroute_env_key = os.environ.get('OMNIROUTE_API_KEY', '')

    # Route through OmniRoute proxy so requests and token analytics show up live in the OmniRoute UI
    provider = 'omniroute'
    api_key = api_key or omniroute_env_key
    model = model or 'mistral/codestral-latest'
    if model in ('agentrouter/gpt-6-astra', 'auto', 'auto/best-coding'):
        model = 'mistral/codestral-latest'
    endpoint = endpoint or 'http://localhost:20128/v1'

    started = research_manager.start_loop(rounds=rounds, provider=provider, api_key=api_key, model=model, endpoint=endpoint, force_restart=force_restart)
    if not started:
        return jsonify({'success': False, 'message': 'Research loop is already running. Click Pause to stop it first, or click Restart.'}), 400
    return jsonify({'success': True, 'message': f'Started Multi-Agent Research Loop ({rounds} rounds)'})


@app.route('/api/research/pause', methods=['POST'])
def api_research_pause():
    paused = research_manager.pause_loop()
    if not paused:
        return jsonify({'success': False, 'message': f'Cannot pause loop in status "{research_manager.status}".'}), 400
    return jsonify({'success': True, 'message': f'Pause signal sent. Loop halting at Round {research_manager.current_round}.'})


@app.route('/api/research/resume', methods=['POST'])
def api_research_resume():
    data = request.get_json(silent=True) or {}
    provider = data.get('provider')
    api_key = data.get('api_key')
    model = data.get('model')
    endpoint = data.get('endpoint')
    target_round = research_manager.current_round + 1 if getattr(research_manager, 'round_completed', True) else max(1, research_manager.current_round)
    resumed = research_manager.resume_loop(provider=provider, api_key=api_key, model=model, endpoint=endpoint)
    if not resumed:
        return jsonify({'success': False, 'message': f'Cannot resume loop in status "{research_manager.status}".'}), 400
    return jsonify({'success': True, 'message': f'Resumed research loop from round {target_round} of {research_manager.max_rounds}.'})


@app.route('/api/research/stop', methods=['POST'])
def api_research_stop():
    research_manager.stop_loop()
    return jsonify({'success': True, 'message': 'Stop signal dispatched to Multi-Agent Research Loop.'})


@app.route('/api/research/status')
def api_research_status():
    return jsonify(research_manager.get_state())


@app.route('/api/research/sentinel')
def api_research_sentinel():
    return jsonify({
        'success': True,
        'sentinel': research_manager.get_sentinel_status()
    })


@app.route('/api/research/override_sentinel', methods=['POST'])
def api_research_override_sentinel():
    res = research_manager.override_sentinel()
    # If the manager was paused due to saturation, attempt auto-resume
    if research_manager.status == 'paused':
        research_manager.resume_loop()
    return jsonify(res)


@app.route('/api/research/set_sentinel_threshold', methods=['POST'])
def api_research_set_sentinel_threshold():
    data = request.get_json(silent=True) or {}
    threshold = int(data.get('threshold', 25))
    new_val = research_manager.set_sentinel_threshold(threshold)
    return jsonify({
        'success': True,
        'threshold': new_val,
        'sentinel': research_manager.get_sentinel_status()
    })


@app.route('/api/engine/status')
def api_engine_status():
    return jsonify({
        'success': True,
        'engine': get_engine_telemetry(),
        'research_active': research_manager.status == 'running',
        'research_status': research_manager.status
    })


@app.route('/api/portfolio/ensemble', methods=['GET'])
def api_portfolio_ensemble():
    try:
        data = get_portfolio_ensemble_data()
        return jsonify({'success': True, 'portfolio': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/portfolio/rebuild', methods=['POST'])
def api_portfolio_rebuild():
    try:
        data = build_portfolio_ensemble()
        return jsonify({'success': True, 'portfolio': data, 'message': 'Portfolio ensemble rebuilt successfully across 2026 MT5 data.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Server Startup
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    _initialise()
    app.run(debug=False, port=5000, host='127.0.0.1', threaded=True)

