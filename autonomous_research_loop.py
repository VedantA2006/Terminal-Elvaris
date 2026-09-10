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
import pandas as pd

from ai_research_agents import IdeaGeneratorAgent, RiskOfficerAgent, CriticPostMortem, ParameterGridSweeper, OptimizerAgent, ALPHA_ARCHETYPES
from strategy_executor import execute_strategy
from monte_carlo import run_monte_carlo
from leaderboard import add_strategy_to_leaderboard, compute_monthly_r_breakdown, compute_rank_score, load_leaderboard
from download_data import load_or_download


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

        self.status = "idle"  # 'idle' | 'running' | 'stopping'
        self.current_round = 0
        self.max_rounds = 100
        self.total_candidates = 0
        self.active_agent = "Idle"
        self.current_hypothesis = ""
        self.best_candidate: Optional[Dict[str, Any]] = None
        self.survivors_added = []
        self.logs: List[Dict[str, Any]] = []

        # Deduplication & Novelty Memory
        self.seen_code_hashes: set = set()
        self.seen_signal_fingerprints: set = set()
        self.recent_hypotheses: List[str] = []

        self._raw_df: Optional[pd.DataFrame] = None

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

    def _ensure_data(self) -> pd.DataFrame:
        if self._raw_df is None:
            self._raw_df = load_or_download()
        return self._raw_df

    def start_loop(self, rounds: int = 100, provider: str = "omniroute", api_key: str = "", model: str = "", endpoint: str = None) -> bool:
        """Starts background exploration loop."""
        with self._lock:
            if self.status == "running":
                return False
            self.status = "running"
            self._should_stop = False
            self.current_round = 0
            self.max_rounds = max(1, min(rounds, 100))
            self.logs = []
            self.survivors_added = []

            # Initialize deduplication registry with existing leaderboard codes
            self.seen_code_hashes.clear()
            self.seen_signal_fingerprints.clear()
            self.recent_hypotheses.clear()
            try:
                for strat in load_leaderboard():
                    c = strat.get('code', '')
                    if c:
                        self.seen_code_hashes.add(_compute_code_hash(c))
            except Exception:
                pass

        self._log("System", f"🚀 Starting Autonomous Exploration Tournament ({self.max_rounds} Rounds across 16 Archetypes)", "info")

        self._thread = threading.Thread(
            target=self._run_tournament,
            args=(provider, api_key, model, endpoint),
            daemon=True
        )
        self._thread.start()
        return True

    def stop_loop(self):
        """Signals the background loop to finish its current round and stop."""
        with self._lock:
            if self.status == "running":
                self.status = "stopping"
                self._should_stop = True
                self._log("System", "🛑 Stop signal received. Finishing active agent step...", "warning")

    def get_state(self) -> Dict[str, Any]:
        """Returns snapshot of current research progress."""
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
                "recent_logs": self.logs[-40:]
            }

    def _run_tournament(self, provider: str, api_key: str, model: str, endpoint: str = None):
        """Background execution loop with deduplication & novelty enforcement."""
        try:
            df = self._ensure_data()
            idea_agent = IdeaGeneratorAgent(provider, api_key, model, endpoint)
            risk_agent = RiskOfficerAgent(provider, api_key, model, endpoint)
            opt_agent = OptimizerAgent(provider, api_key, model, endpoint)

            for r in range(1, self.max_rounds + 1):
                if self._should_stop:
                    break

                with self._lock:
                    self.current_round = r

                archetype_idx = (r - 1) % len(ALPHA_ARCHETYPES)
                arch = ALPHA_ARCHETYPES[archetype_idx]

                # STEP 1: IDEA GENERATOR WITH NOVELTY ENFORCEMENT
                with self._lock:
                    self.active_agent = "Idea Generator"
                    self.current_hypothesis = f"Round {r}: {arch['name']} - {arch['concept']}"
                self._log("💡 Idea Generator", f"Mining combinatorial alpha: [{arch['name']}]", "info")

                proposal = None
                for attempt in range(1, 4):
                    try:
                        proposal = idea_agent.propose_strategy(archetype_idx, recent_hypotheses=self.recent_hypotheses)
                        if proposal and proposal.get("code"):
                            break
                    except Exception as e:
                        self._log("💡 Idea Generator", f"Attempt {attempt}/3 failed ({e}). Retrying in 4s...", "warning")
                        time.sleep(4)

                if not proposal or not proposal.get("code"):
                    self._log("💡 Idea Generator", f"Skipping round {r} due to upstream LLM unavailability.", "error")
                    time.sleep(5)
                    continue

                initial_code = proposal["code"]

                if self._should_stop:
                    break

                # TIER 1 DEDUPLICATION CHECK: Normalized Code Hash
                initial_hash = _compute_code_hash(initial_code)
                if initial_hash in self.seen_code_hashes:
                    self._log(
                        "⚡ Deduplication Guard",
                        f"Duplicate strategy code detected (already evaluated in tournament). Skipping backtest to save compute.",
                        "warning"
                    )
                    time.sleep(1)
                    continue
                self.seen_code_hashes.add(initial_hash)

                # STEP 2: RISK OFFICER AUDIT
                with self._lock:
                    self.active_agent = "Risk Officer"
                self._log("🛡️ Risk Officer", "Auditing proposal for lookahead bias and defensive risk controls...", "info")

                approved, audited_code, audit_notes = risk_agent.audit_and_refine(initial_code, arch["name"])
                for note in audit_notes:
                    self._log("🛡️ Risk Officer", note, "success" if "APPROVED" in note or "PASSED" in note else "warning")

                if not approved:
                    self._log("🛡️ Risk Officer", "Rejected proposal due to unresolvable lookahead bias.", "error")
                    continue

                # Re-check audited code hash in case risk officer injection created a duplicate
                audited_hash = _compute_code_hash(audited_code)
                if audited_hash != initial_hash and audited_hash in self.seen_code_hashes:
                    self._log(
                        "⚡ Deduplication Guard",
                        f"Audited code matches an existing evaluated strategy. Skipping backtest to save compute.",
                        "warning"
                    )
                    time.sleep(1)
                    continue
                self.seen_code_hashes.add(audited_hash)

                # STEP 3: PARAMETER GRID OPTIMIZATION & FAST 1-PASS PRUNE
                with self._lock:
                    self.active_agent = "Backtester"
                self._log("📊 Backtest Engine", f"Simulating on real Dukascopy 5m Gold ({len(df):,} bars) & sweeping parameter space...", "info")

                opt_code, stats, trades, monthly_r = ParameterGridSweeper.sweep_and_optimize(audited_code, df)
                audited_code = opt_code
                self.total_candidates += 1

                if not stats or len(trades) < 5:
                    self._log("📊 Backtest Engine", f"Insufficient trades taken ({len(trades)} trades). Fast-pruning round to save compute.", "warning")
                    continue

                # TIER 2 DEDUPLICATION CHECK: Signal & Trade Footprint
                sig_fp = _compute_signal_fingerprint(trades)
                if sig_fp in self.seen_signal_fingerprints:
                    self._log(
                        "⚡ Deduplication Guard",
                        f"Duplicate signal footprint detected (produces identical trade entries to prior strategy). Skipping backtest to save compute.",
                        "warning"
                    )
                    time.sleep(1)
                    continue
                self.seen_signal_fingerprints.add(sig_fp)

                # Update recent hypotheses window for Idea Generator novelty memory
                self.recent_hypotheses.append(f"{arch['name']}: {arch['concept'][:60]}")
                if len(self.recent_hypotheses) > 8:
                    self.recent_hypotheses.pop(0)

                months_10r = sum(1 for v in monthly_r.values() if v >= 10.0)
                mc_res = run_monte_carlo(trades, num_simulations=1000)

                net_r = round(float(sum(monthly_r.values())), 1) if monthly_r else round(float(stats.get('total_pnl', 0.0) / 1000.0), 1)
                self._log(
                    "📊 Backtest Engine",
                    f"Grid Peak Result: {len(trades)} Trades | WR: {stats.get('win_rate')}% | Return: {net_r:+.1f}R | Months >= 10R: {months_10r}",
                    "info"
                )

                if self._should_stop:
                    break

                # STEP 4: CRITIC POST-MORTEM
                with self._lock:
                    self.active_agent = "Critic Post-Mortem"
                self._log("🔬 Critic Post-Mortem", "Diagnosing failure modes and loss concentrations...", "info")

                post_mortem = CriticPostMortem.analyze_failures(trades, stats)
                self._log("🔬 Critic Post-Mortem", f"Diagnosis: {post_mortem['diagnosis']}", "warning")

                # STEP 5: OPTIMIZER REFINEMENT
                with self._lock:
                    self.active_agent = "Optimizer"
                self._log("⚡ Optimizer Agent", "Applying empirical adjustments to boost monthly consistency...", "info")

                opt_res = opt_agent.optimize_with_postmortem(
                    audited_code, stats, mc_res, post_mortem, monthly_r, iteration=r
                )

                final_code = audited_code
                final_stats = stats
                final_trades = trades
                final_monthly = monthly_r
                final_mc = mc_res
                final_r = net_r

                if opt_res.get("success") and opt_res.get("code"):
                    opt_exec = execute_strategy(opt_res["code"], df)
                    if opt_exec.get("success") and len(opt_exec.get("trades", [])) >= 5:
                        opt_trades = opt_exec["trades"]
                        opt_stats = opt_exec["stats"]
                        opt_monthly = compute_monthly_r_breakdown(opt_trades)
                        opt_r = round(float(sum(opt_monthly.values())), 1) if opt_monthly else round(float(opt_stats.get('total_pnl', 0.0) / 1000.0), 1)

                        if opt_r >= net_r:
                            self._log("⚡ Optimizer Agent", f"Optimization SUCCEEDED: Return improved from {net_r:+.1f}R to {opt_r:+.1f}R!", "success")
                            final_code = opt_res["code"]
                            final_stats = opt_stats
                            final_trades = opt_trades
                            final_monthly = opt_monthly
                            final_mc = run_monte_carlo(opt_trades, num_simulations=1000)
                            final_r = opt_r
                        else:
                            self._log("⚡ Optimizer Agent", f"Optimized variation did not beat baseline ({opt_r:+.1f}R vs {net_r:+.1f}R). Keeping base candidate.", "info")

                # STEP 6: LEADERBOARD QUALIFICATION
                strat_title = f"{arch['name']} (Round #{r})"
                months_ge_10 = sum(1 for v in final_monthly.values() if v >= 10.0)

                # Add to persistent leaderboard if viable
                entry = add_strategy_to_leaderboard(
                    name=strat_title,
                    concept=f"{arch['concept'][:75]}...",
                    code=final_code,
                    stats=final_stats,
                    trades=final_trades,
                    author="Multi-Agent Quant Team"
                )
                self.survivors_added.append(entry.get("id"))

                with self._lock:
                    if self.best_candidate is None or final_r > self.best_candidate.get("total_r", -999):
                        self.best_candidate = {
                            "name": strat_title,
                            "total_r": final_r,
                            "win_rate": final_stats.get("win_rate"),
                            "trades": len(final_trades),
                            "months_ge_10r": months_ge_10,
                            "rank": entry.get("rank")
                        }

                if months_ge_10 >= 5:
                    self._log(
                        "🏆 CHAMPION DETECTED",
                        f"🌟 Strategy '{strat_title}' achieved {months_ge_10} months >= +10R with {final_r:+.1f}R! Ranked #{entry.get('rank')}.",
                        "success"
                    )
                else:
                    self._log(
                        "🏆 Leaderboard",
                        f"Ranked '{strat_title}' as #{entry.get('rank')} on Strategy Leaderboard ({final_r:+.1f}R, {months_ge_10} high-yield months).",
                        "success"
                    )

                time.sleep(1)

        except Exception as e:
            self._log("System", f"Tournament halted due to exception: {e}", "error")
        finally:
            with self._lock:
                self.status = "idle"
                self.active_agent = "Idle"
            self._log("System", f"🏁 Tournament complete ({self.current_round}/{self.max_rounds} strategies evaluated). Loop stopped.", "info")


# Global singleton instance
research_manager = ResearchLoopManager()
