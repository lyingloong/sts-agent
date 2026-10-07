from __future__ import annotations

import unittest

from sts_agent.evaluation.progress import ProgressTracker, aggregate_summaries


def state(act=1, floor=1, screen="NONE", phase="COMBAT", room="MonsterRoom", hp=70, **extra):
    return {"act": act, "floor": floor, "screen_type": screen, "room_phase": phase,
            "room_type": room, "current_hp": hp, **extra}


class ProgressTests(unittest.TestCase):
    def test_death_on_43_with_two_bosses_matches_paper(self):
        tracker = ProgressTracker()
        tracker.observe(state(act=3, floor=43, screen="GAME_OVER", hp=0))
        self.assertEqual(tracker.result()["cleared_floors"], 42)
        self.assertEqual(tracker.result()["boss_acts_defeated"], [1, 2])
        self.assertAlmostEqual(tracker.result()["progress_score"], 42 + 100 / 3)
        self.assertTrue(tracker.result()["score_final"])

    def test_entered_floor_is_not_cleared_floor(self):
        tracker = ProgressTracker()
        tracker.observe(state(floor=10))
        self.assertEqual(tracker.cleared_floors, 9)
        tracker.observe(state(floor=10, phase="COMPLETE"))
        self.assertEqual(tracker.cleared_floors, 10)
        self.assertFalse(tracker.score_final)

    def test_boss_complete_and_reward_do_not_double_count(self):
        tracker = ProgressTracker()
        for _ in range(3):
            tracker.observe(state(floor=17, room="MonsterRoomBoss", phase="COMBAT"))
        self.assertEqual(tracker.boss_acts, set())
        tracker.observe(state(floor=17, room="MonsterRoomBoss", phase="COMPLETE"))
        tracker.observe(state(floor=17, screen="BOSS_REWARD", phase="COMPLETE"))
        tracker.observe(state(act=2, floor=18))
        self.assertEqual(tracker.boss_acts, {1})

    def test_half_dead_and_event_boss_do_not_get_bonus(self):
        tracker = ProgressTracker()
        tracker.observe(state(act=3, floor=50, room="MonsterRoomBoss",
                              combat={"monsters": [{"name": "Awakened One", "hp": 0, "half_dead": True}]}))
        self.assertEqual(tracker.boss_acts, {1, 2})
        tracker.observe(state(act=3, floor=50, room="EventRoom", phase="COMPLETE"))
        self.assertEqual(tracker.boss_acts, {1, 2})

    def test_victory_uses_paper_cap_preserving_raw_engine_floor(self):
        for screen in ("VICTORY", "GAME_OVER"):
            tracker = ProgressTracker()
            tracker.observe(state(act=3, floor=51, screen=screen, screen_state={"victory": True}))
            result = tracker.result()
            self.assertEqual(result["progress_score"], 100)
            self.assertEqual(result["terminal_floor"], 51)
            self.assertEqual(result["cleared_floors"], 50)
            self.assertTrue(tracker.won)
            self.assertFalse(tracker.beat_heart)

    def test_heart_is_separate_from_act3_score(self):
        for victory in (True, False):
            tracker = ProgressTracker()
            tracker.observe(state(act=4, floor=56, screen="GAME_OVER", hp=0,
                                  screen_state={"victory": victory}))
            self.assertEqual(tracker.result()["progress_score"], 100)
            self.assertTrue(tracker.result()["act3_cleared"])
            self.assertEqual(tracker.beat_heart, victory)

    def test_death_does_not_count_stale_complete_flag(self):
        tracker = ProgressTracker()
        tracker.observe(state(floor=10, phase="COMPLETE"))
        tracker.observe(state(floor=10, screen="GAME_OVER", phase="COMPLETE", hp=0))
        self.assertEqual(tracker.cleared_floors, 9)

    def test_missing_invalid_values_are_not_fabricated(self):
        tracker = ProgressTracker()
        tracker.observe({"act": True, "floor": False})
        self.assertIsNone(tracker.result()["progress_score"])
        tracker.observe({"screen_type": "GAME_OVER"})
        self.assertIsNone(tracker.won)
        self.assertFalse(tracker.score_final)

    def test_boundary_snapshots_and_post_terminal_observations_ignored(self):
        tracker = ProgressTracker()
        tracker.observe(state(act=4, floor=56, recorder_snapshot_only=True))
        self.assertIsNone(tracker.cleared_floors)
        tracker.observe(state(floor=2, screen="GAME_OVER", hp=0))
        tracker.observe(state(act=4, floor=56, screen="GAME_OVER", screen_state={"victory": True}))
        self.assertEqual(tracker.result()["progress_score"], 1)
        self.assertFalse(tracker.won)

    def test_complete_marker_without_hp_is_not_boss_proof(self):
        tracker = ProgressTracker()
        tracker.observe(state(floor=17, room="MonsterRoomBoss", phase="COMPLETE", hp=None))
        self.assertEqual(tracker.boss_acts, set())

    def test_empty_aggregate_is_not_zero_win_rate(self):
        result = aggregate_summaries({})
        self.assertEqual(result["eligible_runs"], 0)
        self.assertIsNone(result["progress_score_mean"])
        self.assertIsNone(result["win_rate"])

    def test_aggregate_excludes_partial_human_and_unfinished_but_reports_counts(self):
        def run(score, eligible=True, terminal=True, human=0, partial=False, won=False):
            return {"terminal": terminal, "human_actions": human, "won": won,
                    "run_config": {"partial_run": partial},
                    "evaluation": {"progress_score": score, "autonomous_eligible": eligible,
                                   "boss_acts_defeated": [1, 2, 3] if won else []}}
        result = aggregate_summaries({"death": run(10), "win": run(100, won=True),
                                     "human": run(100, False, human=1),
                                     "partial": run(100, False, partial=True),
                                     "unfinished": run(20, False, terminal=False)})
        self.assertEqual(result["runs_started"], 5)
        self.assertEqual(result["eligible_runs"], 2)
        self.assertEqual(result["unfinished_runs"], 1)
        self.assertEqual(result["human_intervened_runs"], 1)
        self.assertEqual(result["partial_runs"], 1)
        self.assertEqual(result["progress_score_mean"], 55)
        self.assertEqual(result["win_rate"], 0.5)
        self.assertEqual(result["act_clear_rates"]["3"], 0.5)


if __name__ == "__main__":
    unittest.main()
