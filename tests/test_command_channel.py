from __future__ import annotations

import base64
import unittest
from unittest.mock import patch

from sts_agent.runtime.command_channel import CommandChannel, CommandProtocolFailure
from sts_agent.runtime.lifecycle import GameDisconnected


class CommandChannelTests(unittest.TestCase):
    def setUp(self):
        self.channel = CommandChannel()
        self.state = {"recorder_state_id": "s:state-1", "recorder_command_protocol": 1,
                      "ready_for_command": True}
        self.channel.receive(self.state)

    def reply(self, **changes):
        receipt = {"id": self.channel.inflight, "command": self.channel.inflight_command, "status": "accepted", **changes}
        return {"ready_for_command": True, "recorder_command": receipt}

    def test_wire_preserves_command_and_exact_observation_identity(self):
        wire = self.channel.encode("PLAY 2 0", self.state)
        version, identifier, expected, payload = wire.split()
        self.assertEqual((version, identifier, expected), ("AR1", self.channel.inflight, "s:state-1"))
        self.assertEqual(base64.b64decode(payload).decode(), "PLAY 2 0")
        with self.assertRaises(GameDisconnected):
            self.channel.encode("END", self.state)

    def test_old_ready_state_or_unrelated_error_never_acknowledges(self):
        self.channel.encode("PLAY 2 0", self.state)
        for old in (self.state, {"error": "old error"}, self.reply(id="previous"),
                    {**self.reply(), "recorder_snapshot_only": True}):
            self.assertFalse(self.channel.receive(old))
            self.assertIsNotNone(self.channel.inflight)
        self.assertTrue(self.channel.receive(self.reply(execution_status="executed", transaction_id="tx1")))
        self.assertIsNone(self.channel.inflight)
        self.assertEqual(self.channel.confirmation()["receipt"]["transaction_id"], "tx1")

    def test_queue_acceptance_is_not_execution(self):
        self.channel.encode("PLAY 2 0", self.state)
        self.assertFalse(self.channel.receive(self.reply(status="executing")))
        self.assertFalse(self.channel.receive(self.reply(execution_status="pending")))
        self.assertFalse(self.channel.receive({**self.reply(execution_status="executed"), "ready_for_command": False}))
        self.assertTrue(self.channel.receive(self.reply(execution_status="executed")))

    def test_rejection_and_skipped_execution_are_not_success(self):
        for receipt in ({"status": "rejected", "error": "stale"}, {"execution_status": "skipped"},
                        {"execution_status": "cancelled"}):
            self.channel.encode("PLAY 2 0", self.state)
            message = self.reply(**receipt)
            self.assertTrue(self.channel.receive(message))
            self.assertIn("error", message)

    def test_state_query_can_refresh_without_observation(self):
        self.assertIn(" - ", self.channel.encode("STATE", None))
        self.assertTrue(self.channel.receive(self.reply(status="query")))
        with self.assertRaises(GameDisconnected):
            self.channel.encode("END", None)

    def test_rejected_state_without_decoded_command_stops_and_preserves_receipt(self):
        self.channel.encode("STATE", None)
        identifier = self.channel.inflight
        message = self.reply(status="rejected", error="Input byte array has wrong 4-byte ending unit")
        del message["recorder_command"]["command"]
        with self.assertRaisesRegex(CommandProtocolFailure, "protocol failure.*STATE.*not retried"):
            self.channel.receive(message)
        confirmation = self.channel.confirmation()
        self.assertEqual(confirmation["command_id"], identifier)
        self.assertEqual(confirmation["command"], "STATE")
        self.assertEqual(confirmation["receipt"], message["recorder_command"])
        for command in ("STATE", "END"):
            with self.assertRaisesRegex(CommandProtocolFailure, "protocol failure"):
                self.channel.encode(command, self.state)

    def test_legacy_state_error_stops_instead_of_retrying(self):
        channel = CommandChannel()
        channel.encode("STATE", None)
        with self.assertRaisesRegex(CommandProtocolFailure, "protocol failure.*STATE"):
            channel.receive({"error": "Invalid command: state."})
        self.assertEqual(channel.confirmation()["command"], "STATE")

    def test_old_error_cannot_fail_the_current_state_query(self):
        self.channel.encode("STATE", None)
        self.assertFalse(self.channel.receive(self.reply(id="previous", status="rejected", error="old error")))
        self.assertTrue(self.channel.receive(self.reply(status="query")))
        self.assertIsNone(self.channel.inflight)

    def test_legacy_query_keeps_waiting_through_not_ready_and_clears_after_success(self):
        channel = CommandChannel()
        channel.encode("STATE", None)
        self.assertTrue(channel.receive({"ready_for_command": False}))
        self.assertTrue(channel.awaiting_reply)
        self.assertTrue(channel.receive({"ready_for_command": True}))
        self.assertFalse(channel.awaiting_reply)
        self.assertTrue(channel.receive({"error": "later unrelated error"}))
        self.assertIsNone(channel.failure_reason)

    def test_timeout_unknown_no_blind_retry(self):
        with patch("sts_agent.runtime.command_channel.time.monotonic", return_value=100):
            self.channel.encode("END", self.state)
        with patch("sts_agent.runtime.command_channel.time.monotonic", return_value=161):
            with self.assertRaisesRegex(GameDisconnected, "execution unknown, not retried"):
                self.channel.check_deadline()
        self.assertIsNotNone(self.channel.inflight)

    def test_stock_communicationmod_is_explicitly_unverified(self):
        channel = CommandChannel()
        self.assertEqual(channel.encode("END", None), "END")
        self.assertEqual(channel.confirmation()["transport"], "legacy_unverified")

    def test_matching_id_with_wrong_command_fails_closed(self):
        self.channel.encode("END", self.state)
        with self.assertRaisesRegex(GameDisconnected, "mismatched command"):
            self.channel.receive(self.reply(command="PLAY 1"))

    def test_recovery_proof_only_belongs_to_the_current_matched_reply(self):
        from sts_agent.runtime.recovery import CommandRecovery

        self.channel.encode("END", self.state)
        message = self.reply(status="rejected", execution_status="not_executed", error_code="state_changed")
        self.assertTrue(self.channel.receive(message))
        self.assertTrue(CommandRecovery.proves_not_executed(self.channel.confirmation()))
        # An unsolicited error cannot borrow an earlier rejection's proof.
        self.assertTrue(self.channel.receive({"error": "stale_observation"}))
        self.assertFalse(CommandRecovery.proves_not_executed(self.channel.confirmation()))

    def test_rejected_receipt_with_wrong_command_has_unknown_execution(self):
        self.channel.encode("END", self.state)
        with self.assertRaisesRegex(GameDisconnected, "mismatched command.*execution unknown"):
            self.channel.receive(self.reply(command="PLAY 1", status="rejected",
                execution_status="not_executed", error_code="state_changed"))

    def test_success_boundary_is_current_matched_gameplay_not_queries_or_rejections(self):
        self.channel.encode("START IRONCLAD 0", self.state)
        self.assertIsNone(self.channel.accepted_command)
        self.assertFalse(self.channel.receive(self.reply(execution_status="pending")))
        self.assertIsNone(self.channel.accepted_command)
        self.assertTrue(self.channel.receive(self.reply()))
        self.assertEqual(self.channel.accepted_command, "START IRONCLAD 0")
        self.assertTrue(self.channel.receive(self.state))
        self.assertIsNone(self.channel.accepted_command)  # An old success cannot be borrowed.
        self.channel.encode("STATE", self.state)
        self.assertTrue(self.channel.receive(self.reply(status="query")))
        self.assertIsNone(self.channel.accepted_command)
        for receipt in ({"status": "rejected"}, {"execution_status": "skipped"}, {"execution_status": "cancelled"}):
            self.channel.encode("START IRONCLAD 0", self.state)
            self.assertTrue(self.channel.receive(self.reply(**receipt)))
            self.assertIsNone(self.channel.accepted_command)
        self.channel.encode("END", self.state)
        self.assertTrue(self.channel.receive({**self.reply(), "error": "executor failed"}))
        self.assertIsNone(self.channel.accepted_command)

    def test_legacy_success_boundary_requires_local_outstanding_gameplay(self):
        channel = CommandChannel()
        channel.encode("START IRONCLAD 0", None)
        channel.receive({"ready_for_command": False})
        self.assertIsNone(channel.accepted_command)
        channel.receive({"ready_for_command": True})
        self.assertEqual(channel.accepted_command, "START IRONCLAD 0")
        channel.receive({"ready_for_command": True})
        self.assertIsNone(channel.accepted_command)
        channel.encode("STATE", None)
        channel.receive({"ready_for_command": True})
        self.assertIsNone(channel.accepted_command)
        channel.encode("END", None)
        channel.receive({"error": "failed", "ready_for_command": True})
        self.assertIsNone(channel.accepted_command)


if __name__ == "__main__":
    unittest.main()
