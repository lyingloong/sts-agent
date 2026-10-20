from __future__ import annotations

import unittest

from sts_agent.runtime.recovery import CommandRecovery, RecoveryPhase


def rejected_confirmation(code="stale_observation"):
    return {"transport": "correlated", "reply_matched": True, "command_id": "current",
            "command": "CHOOSE 1", "receipt": {"id": "current", "command": "CHOOSE 1",
            "status": "rejected", "execution_status": "not_executed", "error_code": code}}


class CommandRecoveryTests(unittest.TestCase):
    def test_only_explicit_proof_for_the_two_supported_codes_allows_redecision(self):
        for code in ("stale_observation", "state_changed"):
            with self.subTest(code=code):
                recovery = CommandRecovery()
                recovery.command_failed(rejected_confirmation(code), "arbitrary diagnostic text")
                self.assertEqual(recovery.phase, RecoveryPhase.REFRESH_AUTO)
                self.assertEqual(recovery.consecutive_recoveries, 1)
                recovery.observation_refreshed()
                self.assertEqual(recovery.phase, RecoveryPhase.READY)
                self.assertIsNone(recovery.decision_error)

    def test_ambiguous_legacy_missing_id_and_unknown_errors_require_human(self):
        changes = [
            {"transport": "legacy_unverified"}, {"reply_matched": False},
            {"command_id": "old"}, {"receipt": None},
        ]
        receipt_changes = [
            {"id": None}, {"id": ""}, {"id": "old"}, {"command": None},
            {"status": "accepted"}, {"execution_status": None}, {"execution_status": "unknown"},
            {"execution_status": "skipped"}, {"execution_status": "cancelled"},
            {"error_code": "invalid_envelope"}, {"error_code": "duplicate_command"},
            {"error_code": "command_failed"}, {"error_code": None}, {"error_code": {}},
        ]
        confirmations = [rejected_confirmation() | change for change in changes]
        for change in receipt_changes:
            confirmation = rejected_confirmation()
            confirmation["receipt"].update(change)
            confirmations.append(confirmation)
        for confirmation in confirmations:
            with self.subTest(confirmation=confirmation):
                recovery = CommandRecovery()
                recovery.command_failed(confirmation, "stale_observation state_changed")
                self.assertEqual(recovery.phase, RecoveryPhase.REFRESH_HUMAN)
                recovery.observation_refreshed()
                self.assertEqual(recovery.phase, RecoveryPhase.HUMAN_REQUIRED)
                self.assertEqual(recovery.consecutive_recoveries, 0)
                self.assertIsNotNone(recovery.decision_error)

    def test_state_query_is_never_an_automatically_recoverable_game_action(self):
        confirmation = rejected_confirmation()
        confirmation["command"] = confirmation["receipt"]["command"] = "STATE"
        self.assertFalse(CommandRecovery.proves_not_executed(confirmation))

    def test_three_consecutive_recoveries_then_refresh_and_human_handoff(self):
        recovery = CommandRecovery()
        for count in range(1, 4):
            recovery.command_failed(rejected_confirmation(), "changed")
            self.assertEqual(recovery.phase, RecoveryPhase.REFRESH_AUTO)
            recovery.observation_refreshed()
            self.assertEqual(recovery.consecutive_recoveries, count)
        recovery.command_failed(rejected_confirmation(), "changed again")
        self.assertEqual(recovery.phase, RecoveryPhase.REFRESH_EXHAUSTED)
        recovery.observation_refreshed()
        self.assertEqual(recovery.phase, RecoveryPhase.EXHAUSTED)
        self.assertIn("3 次", recovery.decision_error)
        for _ in range(4):
            recovery.request_read_only_refresh()
            recovery.observation_refreshed()
            self.assertEqual(recovery.phase, RecoveryPhase.EXHAUSTED)
            self.assertEqual(recovery.consecutive_recoveries, 3)

    def test_human_read_only_refresh_clears_pause_and_preserves_budget(self):
        recovery = CommandRecovery()
        recovery.command_failed(rejected_confirmation(), "changed")
        recovery.observation_refreshed()
        recovery.command_failed(rejected_confirmation("command_failed"), "unknown execution")
        recovery.observation_refreshed()
        self.assertEqual(recovery.phase, RecoveryPhase.HUMAN_REQUIRED)
        recovery.request_read_only_refresh()
        recovery.observation_refreshed()
        self.assertEqual(recovery.phase, RecoveryPhase.READY)
        self.assertIsNone(recovery.decision_error)
        self.assertEqual(recovery.consecutive_recoveries, 1)

    def test_only_accepted_gameplay_reopens_exhausted_budget(self):
        recovery = CommandRecovery()
        for _ in range(4):
            recovery.command_failed(rejected_confirmation(), "changed")
            recovery.observation_refreshed()
        recovery.gameplay_accepted()
        self.assertEqual(recovery.snapshot(), {"phase": "ready", "consecutive_recoveries": 0,
            "max_consecutive_recoveries": 3, "reason": None})
        recovery.command_failed(rejected_confirmation(), "new failure")
        self.assertEqual(recovery.phase, RecoveryPhase.REFRESH_AUTO)


if __name__ == "__main__":
    unittest.main()
