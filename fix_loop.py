import os

with open('autonomous_research_loop.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update import
content = content.replace(
    "from ai_research_agents import IdeaGeneratorAgent, RiskOfficerAgent, CriticPostMortem, ParameterGridSweeper, OptimizerAgent, ALPHA_ARCHETYPES, synthesize_archetype_code",
    "from ai_research_agents import IdeaGeneratorAgent, RiskOfficerAgent, CriticPostMortem, ParameterGridSweeper, OptimizerAgent, synthesize_archetype_code, load_archetypes, save_archetypes"
)

# 2. Replace usages
content = content.replace("ALPHA_ARCHETYPES", "load_archetypes()")
# Fix multiple calls in same line if needed (not an issue functionally, just minor overhead)

# 3. Add _regenerate_archetypes method
reg_code = """
    def _regenerate_archetypes(self, provider: str, api_key: str, model: str, endpoint: str):
        self._log("⚡ Alpha Sentinel", "Initiating Autonomous Archetype Regeneration via OmniRoute/Astra...", "warning")
        try:
            from ai_generator import call_ai_llm
            import json
            
            top_strats = load_leaderboard()[:5]
            top_concepts = [s.get('concept', '') for s in top_strats]
            
            prompt = f\"\"\"
We are running an autonomous trading AI that has plateaued. The current archetypes have been fully mined and cannot beat the Top 15 threshold anymore.
We need you to invent 15 BRAND NEW, highly exotic, institutional-grade market archetypes for XAUUSD (Gold) 5-minute candles.

Past successful concepts (do not just copy these):
{json.dumps(top_concepts, indent=2)}

Past failure reasons to avoid:
{json.dumps(self.failure_memory, indent=2)}

Think outside the box! Combine volatility, HTF (1H/4H), macro proxies, orderflow, tick absorption, exotic indicators.
Return ONLY valid JSON matching this schema:
[
  {{
    "id": "unique_id_string",
    "name": "Catchy Title",
    "concept": "1-sentence summary",
    "instructions": "Detailed prompt instructions on how to code it"
  }}
]
\"\"\"
            response = call_ai_llm(provider, api_key, model, prompt, endpoint_url=endpoint)
            # parse json
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                new_archs = json.loads(response[start:end])
                if isinstance(new_archs, list) and len(new_archs) > 0:
                    save_archetypes(new_archs)
                    self._log("⚡ Alpha Sentinel", f"Successfully regenerated {len(new_archs)} new archetypes! Resetting plateau counter and continuing loop.", "success")
                    self.rounds_since_top15_beat = 0
                    self.seen_code_hashes.clear()
                    self.seen_signal_fingerprints.clear()
                    return True
            self._log("⚡ Alpha Sentinel", "Failed to parse new archetypes from AI. Retrying next round.", "error")
        except Exception as e:
            self._log("⚡ Alpha Sentinel", f"Error during archetype regeneration: {e}", "error")
        return False
"""

# Insert reg_code before _run_tournament
content = content.replace("    def _run_tournament", reg_code + "\n    def _run_tournament")

# 4. Modify the sentinel trigger in _run_tournament to call regeneration
old_trigger = """
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
"""

new_trigger = """
                    if self.rounds_since_top15_beat >= self.sentinel_threshold and not self.override_saturation:
                        with self._lock:
                            self.active_agent = "Alpha Sentinel (Regenerating Archetypes)"
                        self._log(
                            "⚡ Alpha Sentinel",
                            f"⚠️ ALPHA SATURATION DETECTED: {self.rounds_since_top15_beat}/{self.sentinel_threshold} rounds dry. Auto-triggering AI self-healing archetype regeneration!",
                            "warning"
                        )
                        self._regenerate_archetypes(provider, api_key, model, endpoint)
"""
content = content.replace(old_trigger, new_trigger)

# 5. Change default threshold in __init__ from 25 to 35
content = content.replace("self.sentinel_threshold: int = 25", "self.sentinel_threshold: int = 35")

with open('autonomous_research_loop.py', 'w', encoding='utf-8') as f:
    f.write(content)
