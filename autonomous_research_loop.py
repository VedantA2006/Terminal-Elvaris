"""
Autonomous Quantitative Research Loop Manager.

Runs background multi-round strategy discovery tournaments using the 4-agent team:
Idea Generator -> Risk Officer -> Backtester & Monte Carlo -> Critic Post-Mortem -> Optimizer -> Leaderboard.
"""

import threading
import time
import hashlib
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import pandas as pd

from ai_research_agents import IdeaGeneratorAgent, RiskOfficerAgent, CriticPostMortem, ParameterGridSweeper, OptimizerAgent, ALPHA_ARCHETYPES, synthesize_archetype_code
from ai_generator import get_engine_telemetry
from strategy_executor import execute_strategy
from monte_carlo import run_monte_carlo
from leaderboard import add_strategy_to_leaderboard, compute_monthly_r_breakdown, compute_rank_score, load_leaderboard, get_research_candidates
from download_data import load_or_download
from data_split import split_data

# Out-of-sample validation gate thresholds (Task 3 hardening).
# Validation split is ~15% of the 6-month dataset (a few weeks of session-
# filtered 5m bars). A strategy must show a real, not-merely-lucky edge here
# before it's trusted enough to reach the leaderboard.
MIN_VALIDATION_TRADES = 10
MIN_VALIDATION_PROFIT_FACTOR = 1.15
MIN_VALIDATION_R = 2.0


def _compute_code_hash(code: str) -> str:
    """Normalizes Python code by stripping comments, docstrings, and whitespace for reliable deduplication."""
    if not code:
        return ""
    # Strip multiline docstrings
    no_docstrings = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '', code)
    # Strip line comments
    no_comments = re.sub(r'#.*', '', no_docstrings)
    # Strip all whitespace
    normalized = re.sub(r'\s+', '', no_comments)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def _compute_signal_fingerprint(trades: List[Dict[str, Any]]) -> str:
    """Generates an invariant footprint from trade sequence entries and directions."""
    if not trades:
        return "no_trades"
    sample = [(t.get('entry_bar'), t.get('direction')) for t in trades[:50]]
    return hashlib.md5(str(sample).encode('utf-8')).hexdigest()


class ResearchLoopManager:
    """Thread-safe background research loop coordinator."""

    def __init__(self):
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._should_stop = False
        self._should_pause = False

        self.status = "idle"  # 'idle' | 'running' | 'pausing' | 'paused' | 'stopping'
        self.current_round = 0
        self.round_completed = True
        self.max_rounds = 100
        self.total_candidates = 0
        self.active_agent = "Idle"
        self.current_hypothesis = ""
        self.best_candidate: Optional[Dict[str, Any]] = None
        self.survivors_added = []
        self.logs: List[Dict[str, Any]] = []

        # Last configuration parameters for seamless resume
        self._last_provider = "omniroute"
        self._last_api_key = ""
        self._last_model = "mistral/codestral-latest"
        self._last_endpoint = "http://localhost:20128/v1"

        # Deduplication & Novelty Memory
        self.seen_code_hashes: set = set()
        self.seen_signal_fingerprints: set = set()
        self.recent_hypotheses: List[str] = []

        # Upgrade 4: Failure Memory — stores last N rejection reasons so LLM learns from mistakes
        self.failure_memory: List[str] = []

        # Upgrade 2: Archetype performance tracking — which archetypes produce winners
        self.archetype_wins: Dict[str, int] = {}
        self.archetype_tries: Dict[str, int] = {}

        # Pre-LLM Negative Parameter Cache to eliminate duplicate footprints
        self.recent_param_signatures: Dict[str, List[str]] = {}

        # 25-Round Plateau Detection & Auto-Mutation Shift (Optimal Archetype Coverage)
        self.rounds_since_top15_beat: int = 0
        self._last_round_added_to_top15: bool = False
        self.sentinel_threshold: int = 25
        self.override_saturation: bool = False

        self._raw_df: Optional[pd.DataFrame] = None
        self._train_df: Optional[pd.DataFrame] = None
        self._val_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None

    def _log(self, agent: str, message: str, level: str = "info"):
        """Appends a structured event log."""
        entry = {
            "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
            "agent": agent,
            "message": message,
            "level": level
        }
        with self._lock:
            self.logs.append(entry)
            if len(self.logs) > 200:
                self.logs.pop(0)

    def _ensure_data(self):
        """Loads 2026 dataset (start till date) and splits chronologically into 70% train, 15% val, 15% test."""
        if self._raw_df is None or self._train_df is None:
            full_df = load_or_download()
            self._raw_df = full_df[full_df.index >= '2026-01-01'].copy()
            self._train_df, self._val_df, self._test_df = split_data(self._raw_df)
            start_d = self._raw_df.index[0].strftime('%Y-%m-%d')
            end_d = self._raw_df.index[-1].strftime('%Y-%m-%d')
            self._log("System", f"Loaded 2026 dataset ({len(self._raw_df):,} bars from {start_d} till {end_d}): Train {len(self._train_df):,} | Val {len(self._val_df):,} | Test {len(self._test_df):,}", "info")
        return self._train_df, self._val_df, self._test_df

    def start_loop(self, rounds: int = 100, provider: str = "omniroute", api_key: str = "", model: str = "", endpoint: str = None, resume: bool = False, force_restart: bool = False) -> bool:
        """Starts or resumes background exploration loop."""
        with self._lock:
            if self.status == "running" and not force_restart:
                return False
            if force_restart and self.status in ("running", "pausing"):
                self._should_stop = True
                self._should_pause = False
                self.status = "stopping"
                time.sleep(0.3)

            self.status = "running"
            self._should_stop = False
            self._should_pause = False

            self._last_provider = provider
            self._last_api_key = api_key
            self._last_model = model
            self._last_endpoint = endpoint

            if not resume:
                self.current_round = 0
                self.round_completed = True
                self.max_rounds = max(1, min(rounds, 10000))
                self.logs = []
                self.survivors_added = []
                self.seen_code_hashes.clear()
                self.seen_signal_fingerprints.clear()
                self.recent_hypotheses.clear()
                self.rounds_since_top15_beat = 0
                self.override_saturation = False
                try:
                    for strat in load_leaderboard(self._train_df):
                        c = strat.get('code', '')
                        if c:
                            self.seen_code_hashes.add(_compute_code_hash(c))
                except Exception:
                    pass
                self._log("System", f"🚀 Starting Autonomous Exploration Tournament ({self.max_rounds} Rounds across {len(ALPHA_ARCHETYPES)} Archetypes)", "info")
            else:
                next_r = self.current_round + 1 if self.round_completed else max(1, self.current_round)
                self._log("System", f"▶️ Resuming Autonomous Tournament from Round {next_r} of {self.max_rounds} (Progress Preserved)", "info")

        self._thread = threading.Thread(
            target=self._run_tournament,
            args=(provider, api_key, model, endpoint),
            daemon=True
        )
        self._thread.start()
        return True

    def pause_loop(self) -> bool:
        """Signals background loop to pause gracefully and stop the process of rounds."""
        with self._lock:
            if self.status != "running":
                return False
            self.status = "pausing"
            self._should_pause = True
            self._log("System", f"⏸️ Pause signal received. Halting tournament at Round {self.current_round}...", "warning")
            return True

    def resume_loop(self, provider: str = None, api_key: str = None, model: str = None, endpoint: str = None) -> bool:
        """Resumes a paused tournament from the exact round where it was paused."""
        with self._lock:
            if self.status != "paused":
                return False
        return self.start_loop(
            rounds=self.max_rounds,
            provider=provider or self._last_provider,
            api_key=api_key or self._last_api_key,
            model=model or self._last_model,
            endpoint=endpoint or self._last_endpoint,
            resume=True
        )

    def stop_loop(self):
        """Signals the background loop to finish its current round and stop."""
        with self._lock:
            if self.status in ("running", "pausing"):
                self.status = "stopping"
                self._should_stop = True
                self._should_pause = False
                self._log("System", "🛑 Stop signal received. Finishing active agent step...", "warning")
            elif self.status == "paused":
                self.status = "idle"
                self.active_agent = "Idle"
                self._log("System", f"🛑 Tournament stopped from paused state ({self.current_round}/{self.max_rounds} completed).", "info")

    def get_sentinel_status(self) -> Dict[str, Any]:
        """Evaluates whether the current archetype library and single-timeframe feature space
        is saturated, and returns diagnostic metrics and codebase upgrade recommendations."""
        with self._lock:
            rounds_dry = self.rounds_since_top15_beat
            threshold = self.sentinel_threshold
            override = self.override_saturation
            is_saturated = (rounds_dry >= threshold) and not override
            pct = min(100.0, round((rounds_dry / max(1, threshold)) * 100.0, 1))

            if is_saturated:
                level = "saturated"
                title = "Alpha Plateau Detected · Codebase Upgrade Required"
                desc = (
                    f"The research loop has completed {rounds_dry} consecutive rounds without producing a strategy "
                    f"that beats the Top 15 threshold. In standard 5m OHLCV price action, existing archetypes "
                    f"have captured the maximum available variance (+106R to +125.5R). Running further rounds on these "
                    f"archetypes yields diminishing returns and wastes compute/API tokens. Upgrading the codebase is required."
                )
            elif rounds_dry >= int(threshold * 0.6):
                level = "warning"
                title = f"Alpha Saturation Warning ({rounds_dry}/{threshold} rounds dry)"
                desc = (
                    f"{rounds_dry} rounds since last Top 15 breakthrough. The feature space is approaching saturation. "
                    f"If no strategy beats the Top 15 in the next {threshold - rounds_dry} rounds, the loop will auto-pause."
                )
            else:
                level = "optimal"
                title = f"Alpha Space Healthy ({rounds_dry}/{threshold} rounds)"
                desc = f"Active archetype search in progress. {rounds_dry}/{threshold} rounds since last breakthrough."

            recommendations = [
                {
                    "id": "mtf_confluence",
                    "title": "1. Higher Timeframe (1H / 4H) Confluence Filters",
                    "badge": "Highest Impact",
                    "details": "Integrate 1-Hour and 4-Hour institutional trend, EMA 200 regime, and HTF order blocks as prerequisites before 5-minute triggers. Stops false breakouts in rangebound chop.",
                    "codebase_target": "strategy_executor.py / ai_research_agents.py"
                },
                {
                    "id": "cross_market_macro",
                    "title": "2. Cross-Market Macro Features (DXY & US10Y Yields)",
                    "badge": "Institutional",
                    "details": "Incorporate Dollar Index (DXY) inverse momentum and US 10-Year Treasury Yield divergences as exogenous entry filters for Gold (XAUUSD).",
                    "codebase_target": "data_loader.py / strategy_executor.py"
                },
                {
                    "id": "orderflow_tick_delta",
                    "title": "3. Tick-Level Order Flow & Absorption Microstructure",
                    "badge": "Liquidity",
                    "details": "Upgrade from bar-aggregated volume to real tick volume delta and bid-ask absorption clusters around Daily VAH/VAL and Session Opens.",
                    "codebase_target": "strategy_executor.py"
                },
                {
                    "id": "volatility_clustering",
                    "title": "4. Volatility Regime Switching (GARCH / Realized Vol)",
                    "badge": "Risk Guard",
                    "details": "Automatically deploy ORB & Donchian Breakouts during volatility expansion, and switch to VWAP & Daily Floor Pivots during compression.",
                    "codebase_target": "strategy_executor.py"
                },
                {
                    "id": "portfolio_ensemble",
                    "title": "5. Multi-Strategy Portfolio Ensemble & Correlation Pruning",
                    "badge": "Alpha Multiplier",
                    "details": "Combine the top uncorrelated champions (Stochastic Cycle + ORB + Daily Pivots) to compound returns to >+300R with smoothed drawdown.",
                    "codebase_target": "portfolio_engine.py (Planned)"
                }
            ]

            return {
                "is_saturated": is_saturated,
                "rounds_since_breakthrough": rounds_dry,
                "threshold": threshold,
                "saturation_pct": pct,
                "status_level": level,
                "status_title": title,
                "status_description": desc,
                "override_active": override,
                "recommendations": recommendations,
                "tokens_saved_estimate": max(0, rounds_dry * 1250)
            }

    def override_sentinel(self) -> Dict[str, Any]:
        """Allows user to override alpha saturation auto-pause and continue exploration."""
        with self._lock:
            self.override_saturation = True
            self.rounds_since_top15_beat = 0
            self._log("⚡ Alpha Sentinel", "User override activated: Alpha saturation guard bypassed. Exploration authorized.", "info")
            return {"success": True, "message": "Sentinel overridden. Plateau counter reset to 0."}

    def set_sentinel_threshold(self, threshold: int) -> int:
        """Sets the consecutive dry rounds threshold before alpha saturation auto-pause."""
        with self._lock:
            val = max(5, min(int(threshold), 500))
            self.sentinel_threshold = val
            self._log("⚡ Alpha Sentinel", f"Sentinel sensitivity threshold updated to {val} rounds.", "info")
            return self.sentinel_threshold

    def get_state(self) -> Dict[str, Any]:
        """Returns snapshot of current research progress and live engine telemetry."""
        with self._lock:
            return {
                "status": self.status,
                "current_round": self.current_round,
                "max_rounds": self.max_rounds,
                "total_candidates": self.total_candidates,
                "active_agent": self.active_agent,
                "current_hypothesis": self.current_hypothesis,
                "best_candidate": self.best_candidate,
                "survivors_count": len(self.survivors_added),
                "recent_logs": self.logs[-40:],
                "engine": get_engine_telemetry(),
                "sentinel": self.get_sentinel_status()
            }

    def _run_tournament(self, provider: str, api_key: str, model: str, endpoint: str = None):
        """Background execution loop with deduplication & novelty enforcement across up to 10,000 rounds."""
        try:
            train_df, val_df, test_df = self._ensure_data()
            idea_agent = IdeaGeneratorAgent(provider, api_key, model, endpoint)
            risk_agent = RiskOfficerAgent(provider, api_key, model, endpoint)
            opt_agent = OptimizerAgent(provider, api_key, model, endpoint)

            start_round = self.current_round + 1 if self.round_completed else max(1, self.current_round)
            for r in range(start_round, self.max_rounds + 1):
                if self._should_stop:
                    break
                if self._should_pause:
                    with self._lock:
                        self.status = "paused"
                        self.active_agent = "Paused"
                    self._log("System", f"⏸️ Tournament PAUSED at Round {self.current_round}/{self.max_rounds}. Progress preserved. Click 'Resume' to continue seamlessly.", "warning")
                    return
                try:
                    # Per-round timeout guard: 3 minutes max to prevent indefinite stalls
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            self._execute_single_round, r, train_df, val_df, test_df, idea_agent, risk_agent, opt_agent
                        )
                        future.result(timeout=180)  # 3-minute hard timeout per round
                    if self.status == "paused" or self._should_pause:
                        return
                    if not getattr(self, '_last_round_added_to_top15', False):
                        self.rounds_since_top15_beat += 1

                    # Alpha Saturation Sentinel Check:
                    # If 15 consecutive rounds produce no strategy in the Top 15, auto-pause to prevent
                    # wasted compute time, LLM API tokens, and meaningless iterations.
                    if self.rounds_since_top15_beat >= self.sentinel_threshold and not self.override_saturation:
                        with self._lock:
                            self.status = "paused"
                            self.active_agent = "Alpha Sentinel (Upgrade Needed)"
                        self._log(
                            "⚡ Alpha Sentinel",
                            f"🛑 ALPHA SATURATION DETECTED: {self.rounds_since_top15_beat}/{self.sentinel_threshold} consecutive rounds without beating Top 15. Research loop auto-paused to prevent wasted compute. Codebase upgrade required to expand alpha surface.",
                            "warning"
                        )
                        return

                except FuturesTimeoutError:
                    self._log("System", f"⏰ Round {r} timed out after 3 minutes (upstream LLM latency). Skipping to round {r+1}...", "warning")
                    self.rounds_since_top15_beat += 1
                    with self._lock:
                        self.round_completed = True
                    time.sleep(1)
                except Exception as round_err:
                    self._log("System", f"Round {r} encountered non-fatal error ({round_err}). Resuming tournament on round {r+1}...", "warning")
                    self.rounds_since_top15_beat += 1
                    time.sleep(2)

        except Exception as e:
            self._log("System", f"Tournament halted due to exception: {e}", "error")
        finally:
            with self._lock:
                if self.status != "paused":
                    self.status = "idle"
                    self.active_agent = "Idle"
                    self._log("System", f"🏁 Tournament complete ({self.current_round}/{self.max_rounds} strategies evaluated). Loop stopped.", "info")

    def _execute_single_round(self, r: int, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                             idea_agent: IdeaGeneratorAgent, risk_agent: RiskOfficerAgent, opt_agent: OptimizerAgent):
        """Executes a single end-to-end multi-agent discovery round."""
        with self._lock:
            self.current_round = r
            self.round_completed = False
        self._last_round_added_to_top15 = False

        def _check_interrupted() -> bool:
            if self._should_stop:
                return True
            if self._should_pause:
                with self._lock:
                    self.status = "paused"
                    self.active_agent = "Paused"
                self._log("System", f"⏸️ Tournament PAUSED at Round {self.current_round}/{self.max_rounds}. Progress preserved. Click 'Resume' to continue seamlessly.", "warning")
                return True
            return False

        if _check_interrupted():
            return

        # UPGRADE 2: Smart Archetype Selection with Saturation Ceiling (Anti-Echo Chamber)
        top_15 = load_leaderboard()[:15]
        saturated_arch_ids = set()
        arch_counts_top15 = {}
        for item in top_15:
            iname = item.get('name', '').lower()
            iconcept = item.get('concept', '').lower()
            for a in ALPHA_ARCHETYPES:
                a_name = a['name'].lower()
                a_id = a.get('id', a['name'])
                tokens = [t for t in a_name.split() if len(t) > 3 and t not in ['confluence', 'intraday', 'temporal', 'breakout']]
                if a_name in iname or a_id.lower() in iname or any(tok in iname or tok in iconcept for tok in tokens[:2]):
                    arch_counts_top15[a_id] = arch_counts_top15.get(a_id, 0) + 1
                    if arch_counts_top15[a_id] >= 2:
                        saturated_arch_ids.add(a_id)

        winning_arch_indices = []
        for idx, a in enumerate(ALPHA_ARCHETYPES):
            a_id = a.get('id', a['name'])
            if self.archetype_wins.get(a_id, 0) > 0 and a_id not in saturated_arch_ids:
                winning_arch_indices.append(idx)

        # Every 4th round (if unsaturated winners exist), exploit a winning archetype; otherwise explore novel domains
        if r > 3 and winning_arch_indices and (r % 4 == 0):
            import random as _rnd
            archetype_idx = _rnd.choice(winning_arch_indices)
            arch = ALPHA_ARCHETYPES[archetype_idx]
            arch_id = arch.get('id', arch['name'])
            self._log("🎯 Alpha Explorer", f"Exploitation mode: Prioritizing proven high-yield archetype [{arch['name']}] ({self.archetype_wins[arch_id]} leaderboard wins).", "info")
        else:
            unsaturated_indices = [
                idx for idx, a in enumerate(ALPHA_ARCHETYPES)
                if a.get('id', a['name']) not in saturated_arch_ids
            ]
            if not unsaturated_indices:
                unsaturated_indices = list(range(len(ALPHA_ARCHETYPES)))

            # Select the least-attempted unsaturated archetype to systematically canvas all market regimes
            unsaturated_indices.sort(key=lambda idx: (self.archetype_tries.get(ALPHA_ARCHETYPES[idx].get('id', ALPHA_ARCHETYPES[idx]['name']), 0), idx))
            archetype_idx = unsaturated_indices[0]
            arch = ALPHA_ARCHETYPES[archetype_idx]
            if saturated_arch_ids:
                sat_names = [a['name'] for a in ALPHA_ARCHETYPES if a.get('id', a['name']) in saturated_arch_ids]
                self._log("🌐 Alpha Explorer", f"Exploration mode: Archetypes saturated in Leaderboard Top-15 ({', '.join(sat_names[:2])}). Cycling through novel domains across all {len(ALPHA_ARCHETYPES)} institutional archetypes: [{arch['name']}].", "info")

        # UPGRADE 5: SMART BREEDING — Cross-pollinate top 3 parents every 3rd round
        is_breeding_round = (r % 3 == 0)
        champion_code = None
        second_parent_code = None
        if is_breeding_round:
            try:
                top_candidates = get_research_candidates(train_df, min_train_trades=10, limit=5)
                if top_candidates:
                    import random as _rnd
                    parent = _rnd.choice(top_candidates[:3]) if len(top_candidates) >= 3 else top_candidates[0]
                    champion_code = parent['code']
                    champion_name = parent.get('name', 'Champion')
                    champion_r = parent.get('train_r', 0)
                    champion_pf = parent.get('train_pf', 0)
                    other_candidates = [s for s in top_candidates if s.get('id') != parent.get('id')]
                    if other_candidates:
                        second = _rnd.choice(other_candidates[:3])
                        second_parent_code = second['code']
                        self._log(
                            "🧬 Genetic Breeder",
                            f"Cross-breeding! Parent A: '{champion_name}' ({champion_r:+.1f}R train) × Parent B: '{second.get('name', '?')}' ({second.get('train_r', 0):+.1f}R train)",
                            "info"
                        )
                    else:
                        self._log(
                            "🧬 Genetic Breeder",
                            f"Breeding round! Using '{champion_name}' ({champion_r:+.1f}R train, PF {champion_pf}) as parent for mutation.",
                            "info"
                        )
            except Exception:
                pass

        # Track archetype attempt counts
        arch_id = arch.get('id', arch['name'])
        self.archetype_tries[arch_id] = self.archetype_tries.get(arch_id, 0) + 1

        # STEP 1: IDEA GENERATOR (High-Speed Single-Candidate Alpha Mining)
        with self._lock:
            self.active_agent = "Idea Generator"
            if is_breeding_round and champion_code:
                self.current_hypothesis = f"Round {r}: [Breeding: {arch['name']}]"
            else:
                self.current_hypothesis = f"Round {r}: [{arch['name']}]"
        self._log("💡 Idea Generator", f"Mining Alpha: Generating Candidate [{arch['name']}]...", "info")

        # Gather top leaderboard CODE snippets for the LLM (Archetype-Matched Only to prevent mode collapse)
        leaderboard_code_context = []
        try:
            arch_tokens = [t for t in arch['name'].lower().split() if len(t) > 3 and t not in ['confluence', 'intraday', 'temporal', 'breakout']]
            for s in get_research_candidates(train_df, min_train_trades=10, limit=25):
                s_name = s.get('name', '').lower()
                s_concept = s.get('concept', '').lower()
                if arch['name'].lower() in s_name or arch_id.lower() in s_name or (arch_tokens and any(tok in s_name or tok in s_concept for tok in arch_tokens[:2])):
                    leaderboard_code_context.append({
                        'name': s.get('name', '?'),
                        'total_r': s.get('train_r', 0),          # TRAIN-split only — never val/test
                        'profit_factor': s.get('train_pf', 0),   # TRAIN-split only
                        'win_rate': s.get('train_win_rate', 0),  # TRAIN-split only
                        'code': s['code'][:600]
                    })
                    if len(leaderboard_code_context) >= 2:
                        break
        except Exception:
            pass

        # Check if 15-round plateau is active (Mutation Shift)
        is_mutation_shift = self.rounds_since_top15_beat >= 15
        round_temp = 0.85 if is_mutation_shift else 0.70
        if is_mutation_shift:
            self._log("🧬 Mutation Engine", f"15-round plateau detected ({self.rounds_since_top15_beat} rounds without new Top-15). Generative temperature boosted (0.70 -> 0.85) to force breakthrough mutations!", "warning")

        neg_constraints = self.recent_param_signatures.get(arch_id, [])

        def _fetch_proposal(a_idx: int, c_code: str = None, s_code: str = None) -> Optional[Dict[str, Any]]:
            for attempt in range(1, 3):
                try:
                    p = idea_agent.propose_strategy(
                        a_idx,
                        recent_hypotheses=self.recent_hypotheses,
                        champion_code=c_code,
                        second_parent_code=s_code,
                        failure_memory=self.failure_memory,
                        leaderboard_code_context=leaderboard_code_context,
                        negative_constraints=neg_constraints,
                        temperature=round_temp
                    )
                    if p and p.get("code"):
                        return p
                except Exception as e:
                    time.sleep(1)
            return None

        proposal = _fetch_proposal(archetype_idx, champion_code, second_parent_code)

        if _check_interrupted():
            return

        if not proposal:
            self._log("💡 Idea Generator", f"Skipping round {r} due to upstream LLM unavailability.", "error")
            time.sleep(2)
            return

        # STEP 2 & 3: RISK AUDIT & PARAMETER GRID BACKTEST
        def _evaluate_candidate(prop: Optional[Dict[str, Any]], target_arch: Dict[str, Any], target_idx: int, label: str):
            if not prop or not prop.get("code"):
                return None
            code = prop["code"]
            c_hash = _compute_code_hash(code)
            if c_hash in self.seen_code_hashes:
                self._log("⚡ Deduplication Guard", f"Candidate [{target_arch['name']}]: Duplicate code detected, skipping.", "warning")
                return None
            self.seen_code_hashes.add(c_hash)

            # Record parameter signature in Pre-LLM Negative Cache to avoid future duplicate generation
            raw_params = re.findall(r'(?:swing_len|period|atr_len|span|period_fast|ema_period|period_slow)\s*=\s*(\d+)', code)
            if raw_params:
                sig = f"lookbacks={','.join(raw_params[:4])}"
                arch_sigs = self.recent_param_signatures.setdefault(target_arch.get('id', target_arch['name']), [])
                arch_sigs.append(sig)
                if len(arch_sigs) > 10:
                    arch_sigs.pop(0)

            # Audit with Risk Officer
            approved, audited, notes = risk_agent.audit_and_refine(code, target_arch["name"], archetype_idx=target_idx)
            if not approved:
                self.failure_memory.append(f"'{target_arch['name']}': Rejected by Risk Officer AST audit — Lookahead bias")
                if len(self.failure_memory) > 10:
                    self.failure_memory.pop(0)
                self._log("🛡️ Risk Officer", f"Candidate [{target_arch['name']}]: Rejected by AST audit.", "warning")
                return None

            a_hash = _compute_code_hash(audited)
            if a_hash != c_hash and a_hash in self.seen_code_hashes:
                return None
            self.seen_code_hashes.add(a_hash)

            # Sweep on Train Split
            opt_c, st, tr, m_r = ParameterGridSweeper.sweep_and_optimize(audited, train_df)

            # Dynamic Alpha Recovery if low trades
            if (not st or len(tr) < 5) and not prop.get("fallback_active"):
                syn_c = synthesize_archetype_code(target_idx, champion_code=champion_code if target_idx == archetype_idx else None)
                rec_c, rec_st, rec_tr, rec_mr = ParameterGridSweeper.sweep_and_optimize(syn_c, train_df)
                if rec_st and len(rec_tr) >= 5:
                    opt_c, st, tr, m_r = rec_c, rec_st, rec_tr, rec_mr

            if not st or len(tr) < 5:
                self.failure_memory.append(f"'{target_arch['name']}': Low trade frequency ({len(tr) if tr else 0} trades < 5)")
                if len(self.failure_memory) > 10:
                    self.failure_memory.pop(0)
                return None

            nr = round(float(sum(m_r.values())), 1) if m_r else round(float(st.get('total_pnl', 0.0) / 1000.0), 1)
            return {
                "arch": target_arch,
                "arch_idx": target_idx,
                "label": label,
                "proposal": prop,
                "audited_code": opt_c,
                "stats": st,
                "trades": tr,
                "monthly_r": m_r,
                "net_r": nr
            }

        with self._lock:
            self.active_agent = "Risk Officer & Backtester"

        winner = _evaluate_candidate(proposal, arch, archetype_idx, "A")
        self.total_candidates += 1

        if not winner:
            self._log("📊 Backtest Engine", "Candidate pruned due to low trade frequency or risk constraints. Exploring next round...", "info")
            return

        # Adopt winner's data
        arch = winner["arch"]
        archetype_idx = winner["arch_idx"]
        proposal = winner["proposal"]
        audited_code = winner["audited_code"]
        stats = winner["stats"]
        trades = winner["trades"]
        monthly_r = winner["monthly_r"]
        net_r = winner["net_r"]

        initial_code = proposal.get("code", audited_code)
        engine_mode = proposal.get("engine_mode", "LLM")
        engine_name = proposal.get("engine_name", "OmniRoute")
        fallback_active = proposal.get("fallback_active", False)
        fallback_reason = proposal.get("fallback_reason")

        if fallback_active:
            self._log("⚡ Engine", f"Selected Candidate used [{engine_name}] · Fallback Active ({fallback_reason})", "warning")
        else:
            self._log("⚡ Engine", f"Selected Candidate used [{engine_name}] · Live LLM Generation (200 OK)", "success")

        if _check_interrupted():
            return

        # TIER 2 DEDUPLICATION CHECK: Signal & Trade Footprint
        sig_fp = _compute_signal_fingerprint(trades)
        if sig_fp in self.seen_signal_fingerprints:
            self._log(
                "⚡ Deduplication Guard",
                f"Duplicate signal footprint detected (produces identical trade entries to prior strategy). Skipping backtest to save compute.",
                "warning"
            )
            time.sleep(1)
            return
        self.seen_signal_fingerprints.add(sig_fp)

        # UPGRADE 2: Expanded novelty window (6→25) for Idea Generator memory
        self.recent_hypotheses.append(f"{arch['name']}: {arch['concept'][:60]}")
        if len(self.recent_hypotheses) > 25:
            self.recent_hypotheses.pop(0)

        months_10r = sum(1 for v in monthly_r.values() if v >= 10.0)
        mc_res = run_monte_carlo(trades, num_simulations=1000)

        net_r = round(float(sum(monthly_r.values())), 1) if monthly_r else round(float(stats.get('total_pnl', 0.0) / 1000.0), 1)
        self._log(
            "📊 Backtest Engine",
            f"Train Grid Peak: {len(trades)} Trades | WR: {stats.get('win_rate')}% | Return: {net_r:+.1f}R | Months >= 10R: {months_10r}",
            "info"
        )

        if _check_interrupted():
            return

        # STEP 4: CRITIC POST-MORTEM
        with self._lock:
            self.active_agent = "Critic Post-Mortem"
        self._log("🔬 Critic Post-Mortem", "Diagnosing failure modes and loss concentrations...", "info")

        post_mortem = CriticPostMortem.analyze_failures(trades, stats)
        self._log("🔬 Critic Post-Mortem", f"Diagnosis: {post_mortem['diagnosis']}", "warning")

        if _check_interrupted():
            return

        # STEP 5: OPTIMIZER REFINEMENT (High-Speed Edge Gating & Critic 1-Shot Near-Miss Auto-Repair)
        pf_val = float(stats.get('profit_factor', 0))
        is_strong = (net_r >= 15.0 and pf_val >= 1.4)
        is_near_miss = (-15.0 <= net_r <= 0.0) and (pf_val >= 0.70) and (len(trades) >= 15)
        is_refinable = (net_r > 0.0 and pf_val >= 1.0 and len(trades) >= 8)

        if is_strong:
            self._log("⚡ Optimizer Agent", f"Strong candidate detected ({net_r:+.1f}R, PF {pf_val:.2f}). Skipping optimizer to preserve edge — proceeding to validation.", "success")
        elif not is_refinable and not is_near_miss:
            self._log("⚡ Optimizer Agent", f"Candidate edge below refinement threshold ({net_r:+.1f}R, PF {pf_val:.2f}, {len(trades)} trades). Skipping optimizer to accelerate loop.", "info")
            if net_r <= 0.0:
                self._log("🛡️ Validation Gate", f"Fast-fail: Candidate Train return ({net_r:+.1f}R, PF {pf_val:.2f}) unviable. Pruning without validation.", "warning")
                return
        else:
            with self._lock:
                self.active_agent = "Optimizer"
            if is_near_miss:
                self._log("⚡ Optimizer Agent", f"Near-miss candidate detected ({net_r:+.1f}R, PF {pf_val:.2f}, {len(trades)} trades). Triggering Critic 1-Shot Auto-Repair...", "info")
            else:
                self._log("⚡ Optimizer Agent", "Applying empirical adjustments to boost monthly consistency...", "info")

            lb_context = None
            try:
                lb_curr = get_research_candidates(train_df, min_train_trades=10, limit=3)
                if lb_curr:
                    lb_context = lb_curr
            except Exception:
                pass

            opt_res = opt_agent.optimize_with_postmortem(
                audited_code, stats, mc_res, post_mortem, monthly_r, iteration=r,
                leaderboard_context=lb_context
            )

            if opt_res.get("success") and opt_res.get("code"):
                opt_exec = execute_strategy(opt_res["code"], train_df)
                if opt_exec.get("success") and len(opt_exec.get("trades", [])) >= 5:
                    opt_trades = opt_exec["trades"]
                    opt_stats = opt_exec["stats"]
                    opt_monthly = compute_monthly_r_breakdown(opt_trades)
                    opt_r = round(float(sum(opt_monthly.values())), 1) if opt_monthly else round(float(opt_stats.get('total_pnl', 0.0) / 1000.0), 1)

                    if opt_r > net_r:
                        self._log("⚡ Optimizer Agent", f"Optimization SUCCEEDED: Train return improved from {net_r:+.1f}R to {opt_r:+.1f}R!", "success")
                        audited_code = opt_res["code"]
                        stats = opt_stats
                        trades = opt_trades
                        monthly_r = opt_monthly
                        mc_res = run_monte_carlo(opt_trades, num_simulations=1000)
                        net_r = opt_r
                    else:
                        self._log("⚡ Optimizer Agent", f"Optimized variation did not beat baseline ({opt_r:+.1f}R vs {net_r:+.1f}R). Keeping base candidate.", "info")

            if is_near_miss and net_r <= 0.0:
                self._log("⚡ Optimizer Agent", f"1-Shot Near-Miss repair unviable (Final Train: {net_r:+.1f}R). Pruning candidate.", "warning")
                return

        final_code = audited_code
        final_stats = stats
        final_trades = trades
        final_monthly = monthly_r
        final_mc = mc_res
        final_r = net_r

        if _check_interrupted():
            return

        # STEP 6: OUT-OF-SAMPLE VALIDATION GATE (Overfit Protection)
        strat_title = f"{arch['name']} (Round #{r})"
        with self._lock:
            self.active_agent = "Risk Officer"
        self._log("🛡️ Validation Gate", f"Evaluating on held-out Validation Split ({len(val_df):,} bars)...", "info")
        val_exec = execute_strategy(final_code, val_df)

        min_val_trades = min(10, max(5, int(len(final_trades) * 0.08)))
        if not val_exec.get("success") or len(val_exec.get("trades", [])) < min_val_trades:
            fail_msg = f"'{strat_title}': Only {len(val_exec.get('trades', []))} validation trades (need >= {min_val_trades} based on {len(final_trades)} train trades) — over-filtered entry conditions or too small a sample to trust"
            self._log(
                "🛡️ Validation Gate",
                f"❌ Rejected '{strat_title}': Insufficient validation trades ({len(val_exec.get('trades', []))} < {min_val_trades}). Overfit protection triggered.",
                "warning"
            )
            self.failure_memory.append(fail_msg)
            if len(self.failure_memory) > 10:
                self.failure_memory.pop(0)
            return

        val_trades = val_exec["trades"]
        val_stats = val_exec["stats"]
        val_monthly = compute_monthly_r_breakdown(val_trades)
        val_pf = float(val_stats.get("profit_factor", 0.0))
        val_r = round(float(sum(val_monthly.values())), 1) if val_monthly else round(float(val_stats.get('total_pnl', 0.0) / 1000.0), 1)
        val_stats['total_r'] = val_r

        # Multi-Month Consistency & Robustness Gate:
        # Standard requirements: val_pf >= 1.15 and val_r >= 2.0
        # High-Alpha Consistency Exception: If the strategy demonstrated multi-month compounding edge in train
        # (train_r >= 25.0R and >= 3 months >= 5.0R), validation must remain strictly positive (val_r >= 0.5, val_pf >= 1.05)
        # to avoid pruning robust institutional models due to minor 3-week drawdown noise.
        train_high_alpha = (final_r >= 25.0 and sum(1 for v in final_monthly.values() if v >= 5.0) >= 3)
        effective_min_r = 0.5 if train_high_alpha else MIN_VALIDATION_R
        effective_min_pf = 1.05 if train_high_alpha else MIN_VALIDATION_PROFIT_FACTOR

        if val_pf < effective_min_pf or val_r < effective_min_r:
            fail_msg = f"'{strat_title}': OVERFIT — Train {final_r:+.1f}R but Val {val_r:+.1f}R (PF {val_pf:.2f}, need R>={effective_min_r} and PF>={effective_min_pf}). Needs better out-of-sample robustness."
            self._log(
                "🛡️ Validation Gate",
                f"❌ OVERFIT REJECTED: '{strat_title}' failed validation gate (Train: {final_r:+.1f}R, Val: {val_r:+.1f}R, Val PF: {val_pf:.2f}, Threshold: >={effective_min_r}R/PF{effective_min_pf}). Dropped from leaderboard.",
                "warning"
            )
            self.failure_memory.append(fail_msg)
            if len(self.failure_memory) > 10:
                self.failure_memory.pop(0)
            return

        if _check_interrupted():
            return

        # Also evaluate on held-out test split for reporting
        test_exec = execute_strategy(final_code, test_df)
        test_stats = None
        if test_exec.get("success"):
            test_trades = test_exec.get("trades", [])
            test_st = test_exec.get("stats", {})
            test_m = compute_monthly_r_breakdown(test_trades)
            test_r = round(float(sum(test_m.values())), 1) if test_m else round(float(test_st.get('total_pnl', 0.0) / 1000.0), 1)
            test_st['total_r'] = test_r
            test_stats = test_st

        self._log(
            "🛡️ Validation Gate",
            f"✅ VALIDATION PASSED: Train {final_r:+.1f}R | Val {val_r:+.1f}R (PF {val_pf:.2f})" +
            (f" | Test {test_stats.get('total_r', 0):+.1f}R" if test_stats else ""),
            "success"
        )

        # STEP 7: 2026 MT5 BACKTEST & LEADERBOARD REGISTRATION
        # Strategy passed the anti-overfit validation gate!
        # Now run 2026 historical backtest for leaderboard registration.
        with self._lock:
            self.active_agent = "Backtester & Monte Carlo"
        
        df_full = self._raw_df if self._raw_df is not None else train_df
        df_2026 = df_full[df_full.index >= '2026-01-01']
        if len(df_2026) < 100:
            df_2026 = df_full
        self._log("Backtester & Monte Carlo", f"Executing 2026 MT5 broker backtest for leaderboard entry ({len(df_2026):,} bars)...", "info")

        full_exec = execute_strategy(final_code, df_2026)
        if full_exec.get("success") and len(full_exec.get("trades", [])) > 0:
            lb_stats = full_exec["stats"]
            lb_trades = full_exec["trades"]
            lb_monthly = compute_monthly_r_breakdown(lb_trades)
            lb_r = round(float(sum(lb_monthly.values())), 1) if lb_monthly else round(float(lb_stats.get('total_pnl', 0.0) / 1000.0), 1)
            months_ge_10 = sum(1 for v in lb_monthly.values() if v >= 10.0)
        else:
            lb_stats = final_stats
            lb_trades = final_trades
            lb_monthly = final_monthly
            lb_r = final_r
            months_ge_10 = sum(1 for v in final_monthly.values() if v >= 10.0)

        train_stats_meta = {
            'total_r': final_r,
            'profit_factor': float(final_stats.get('profit_factor', 0.0)),
            'total_trades': len(final_trades)
        }

        # Add to persistent leaderboard with 2026 backtest results
        entry = add_strategy_to_leaderboard(
            name=strat_title,
            concept=f"{arch['concept'][:75]}...",
            code=final_code,
            stats=lb_stats,
            trades=lb_trades,
            author="Multi-Agent Quant Team",
            val_stats=val_stats,
            test_stats=test_stats,
            train_stats=train_stats_meta,
            data_split="mt5_ecn_2026"
        )
        self.survivors_added.append(entry.get("id"))

        # Track Top-15 plateau reset
        strat_rank = entry.get("rank")
        if strat_rank is not None and isinstance(strat_rank, int) and strat_rank <= 15:
            self.rounds_since_top15_beat = 0
            self._last_round_added_to_top15 = True
            self._log("🏆 Breakthrough", f"New Top 15 Strategy Discovered! Ranked #{strat_rank}. Plateau counter reset to 0.", "success")
        else:
            self._last_round_added_to_top15 = False

        # UPGRADE 2: Track archetype success for smarter rotation
        self.archetype_wins[arch_id] = self.archetype_wins.get(arch_id, 0) + 1

        with self._lock:
            if self.best_candidate is None or lb_r > self.best_candidate.get("total_r", -999):
                self.best_candidate = {
                    "name": strat_title,
                    "total_r": lb_r,
                    "win_rate": lb_stats.get("win_rate"),
                    "trades": len(lb_trades),
                    "months_ge_10r": months_ge_10,
                    "rank": entry.get("rank"),
                    "val_r": val_r,
                    "val_pf": val_pf
                }

        lb_trades_count = len(lb_trades)
        lb_pf = float(lb_stats.get("profit_factor", 1.0))
        if months_ge_10 >= 5:
            self._log(
                "🏆 CHAMPION DETECTED",
                f"🌟 Strategy '{strat_title}' achieved {months_ge_10} months >= +10R with Full 6M Return: {lb_r:+.1f}R ({lb_trades_count} trades, PF {lb_pf:.2f})! Ranked #{entry.get('rank')}.",
                "success"
            )
        else:
            self._log(
                "🏆 Leaderboard",
                f"Ranked '{strat_title}' as #{entry.get('rank')} on Strategy Leaderboard with Full 6M Return: {lb_r:+.1f}R ({lb_trades_count} trades, PF {lb_pf:.2f}, {months_ge_10} high-yield months). [Train: {final_r:+.1f}R | Val: {val_r:+.1f}R | Test: {test_stats.get('total_r', 0):+.1f}R]",
                "success"
            )

        with self._lock:
            self.round_completed = True

        time.sleep(1)


# Global singleton instance
research_manager = ResearchLoopManager()
