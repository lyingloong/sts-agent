import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sts_agent.commands.check_config import main
from sts_agent.core.modes import MODE_FRAMEWORKS
from sts_agent.paths import PROJECT_ROOT


class CheckConfigTests(unittest.TestCase):
    def run_command(self, *args):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(list(args))
        return status, stdout.getvalue(), stderr.getvalue()

    def test_list_works_without_local_files(self):
        status, output, _ = self.run_command("--config", "absent.json", "--list")
        self.assertEqual(status, 0)
        self.assertEqual(output.splitlines(), sorted(MODE_FRAMEWORKS))

    def test_check_all_reports_incomplete_profiles_and_checks_the_rest(self):
        example = PROJECT_ROOT / "config" / "agent.example.json"
        status, output, error = self.run_command("--config", str(example), "--all")
        self.assertEqual(status, 1)
        self.assertIn("online_rl:", error)
        self.assertIn("initial_adapter", error)
        self.assertEqual(output.count("collect_human:"), 1)
        self.assertIn("teacher_sft:", output)

    def test_all_profiles_validate_with_a_configured_adapter(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "local.json"
            settings = {
                "schema_version": 2,
                "active_mode": "collect_human",
                "mode_overrides": {"online_rl": {"policy": {"model": {"initial_adapter": "private-adapter"}}}},
            }
            path.write_text(json.dumps(settings), encoding="utf-8")
            status, output, error = self.run_command("--config", str(path), "--all")
            self.assertEqual(status, 0, error)
            self.assertNotIn("private-adapter", output)
            for name in MODE_FRAMEWORKS:
                self.assertIn(f"{name}:", output)

    def test_overriding_mode_does_not_write_to_config(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "local.json"
            original = '{"schema_version": 2, "active_mode": "collect_human"}'
            path.write_text(original, encoding="utf-8")
            status, output, error = self.run_command("--config", str(path), "--mode", "baseline_api")
            self.assertEqual(status, 0, error)
            self.assertIn("baseline_api:", output)
            self.assertEqual(path.read_text(encoding="utf-8"), original)
