"""Orak paper-style three-act progress score for standard StS1 runs.

Use actual room completion/act transitions, never monster HP or boss names.
This is an evaluation metric, not a training reward or a reproduction of the
Orak rule-agent protocol. Raw engine floors remain separately available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev

METRIC_VERSION = "orak_style_act3_v1"
FLOOR_CAP = 50
BOSS_BONUS = 50 / 3


def integer(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


@dataclass
class ProgressTracker:
    cleared_floors: int | None = None
    boss_acts: set[int] = field(default_factory=set)
    highest_floor: int | None = None
    highest_act: int | None = None
    terminal: bool = False
    terminal_floor: int | None = None
    score_final: bool = False
    won: bool | None = None
    beat_heart: bool | None = None

    def observe(self, state: dict) -> None:
        if self.terminal or not state or state.get("recorder_snapshot_only") is True:
            return
        act, floor = integer(state.get("act")), integer(state.get("floor"))
        screen = state.get("screen_type")
        screen_state = state.get("screen_state")
        screen_state = screen_state if isinstance(screen_state, dict) else {}
        terminal = screen == "GAME_OVER" or (screen == "VICTORY" and act is not None and act >= 3)
        victory = (screen == "VICTORY" or screen_state.get("victory") is True) and act is not None and act >= 3
        if act is not None and 1 <= act <= 4:
            self.highest_act = max(self.highest_act or 0, act)
            # Entering a later act proves the previous act-end bosses passed,
            # including when this controller attached part-way through a run.
            self.boss_acts.update(range(1, act))
            alive = integer(state.get("current_hp"))
            if alive != 0 and (screen == "BOSS_REWARD" or (
                    state.get("room_type") == "MonsterRoomBoss"
                    and state.get("room_phase") == "COMPLETE" and alive is not None)):
                if act <= 3:
                    self.boss_acts.add(act)
        if floor is not None:
            self.highest_floor = max(self.highest_floor or 0, floor)
            # Death on N contributes N-1, even if room_phase is stale COMPLETE.
            cleared = max(0, floor - 1)
            if not terminal and state.get("room_phase") == "COMPLETE" and state.get("current_hp") != 0:
                cleared = floor
            self.cleared_floors = max(self.cleared_floors or 0, min(FLOOR_CAP, cleared))
        if victory or (act is not None and act >= 4):
            self.boss_acts.update((1, 2, 3))
            self.cleared_floors = FLOOR_CAP
        if terminal:
            if not victory and act is not None and act < 4 and floor is not None:
                self.cleared_floors = min(FLOOR_CAP, max(0, floor - 1))
            self.terminal = True
            self.terminal_floor = floor
            self.won = victory if act is not None else None
            self.beat_heart = (victory and act >= 4) if act is not None else None
            self.score_final = (victory or (floor is not None and act is not None and 1 <= act <= 4))

    def result(self) -> dict:
        score = None if self.cleared_floors is None else self.cleared_floors + BOSS_BONUS * len(self.boss_acts)
        return {"metric_version": METRIC_VERSION, "cleared_floors": self.cleared_floors,
                "bosses_defeated": len(self.boss_acts), "boss_acts_defeated": sorted(self.boss_acts),
                "progress_score": score, "score_final": self.score_final,
                "highest_floor": self.highest_floor, "highest_act": self.highest_act,
                "terminal_floor": self.terminal_floor, "act3_cleared": 3 in self.boss_acts}


def aggregate_summaries(summaries: dict[str, dict]) -> dict:
    runs = list(summaries.values())
    eligible = [run for run in runs if (run.get("evaluation") or {}).get("autonomous_eligible")]
    scores = [run["evaluation"]["progress_score"] for run in eligible]
    count = len(eligible)
    return {"metric_version": METRIC_VERSION,
            "aggregation_scope": "terminal, final score, no human action, not partial_run",
            "runs_started": len(runs), "terminal_runs": sum(bool(run.get("terminal")) for run in runs),
            "unfinished_runs": sum(not run.get("terminal") for run in runs),
            "human_intervened_runs": sum(run.get("human_actions", 0) > 0 for run in runs),
            "partial_runs": sum(bool((run.get("run_config") or {}).get("partial_run")) for run in runs),
            "eligible_runs": count, "progress_score_mean": mean(scores) if count else None,
            "progress_score_std": stdev(scores) if count > 1 else 0.0 if count else None,
            "act_clear_rates": {str(act): sum(act in run["evaluation"]["boss_acts_defeated"] for run in eligible) / count
                                if count else None for act in (1, 2, 3)},
            "win_rate": sum(run.get("won") is True for run in eligible) / count if count else None,
            "heart_win_rate": sum(run.get("beat_heart") is True for run in eligible) / count if count else None}
