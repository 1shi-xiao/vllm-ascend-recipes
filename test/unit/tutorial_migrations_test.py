"""CPU regression checks for the public MiniMax launcher embedded in recipes."""

from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "models/en/MiniMaxAI/MiniMax-M2.7.yaml"


def launcher_source(recipe):
    preparation = next(item["content"] for item in recipe["prerequisites"]
                       if "launch_online_dp.py <<'PY'" in item["content"])
    return re.search(r"launch_online_dp.py <<'PY'\n(.*?)\nPY\n", preparation, re.S)[1]


class EmbeddedLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
        cls.source = launcher_source(cls.recipe)

    def exercise(self, *, failure=None, interrupt=False, resist_term=False):
        """Schedule fake OS processes to expose waits that hide peer failures."""
        created, groups, handlers, polled = [], [], {}, []
        test = self

        class Process:
            def __init__(self, command, *, start_new_session):
                test.assertTrue(start_new_session)
                self.command = command
                self.pid = 31000 + len(created)
                self.returncode = None
                created.append(self)

            def poll(self):
                test.assertEqual(len(created), 2, "all local ranks must start before polling")
                polled.append(self.pid)
                if failure and self.pid == 31001:
                    self.returncode = failure
                return self.returncode

            def wait(self, timeout=None):
                test.assertEqual(len(created), 2, "joining rank zero must not block peer startup")
                if self.returncode is None and timeout is not None:
                    raise subprocess.TimeoutExpired(self.command, timeout)
                test.assertIsNotNone(self.returncode, "must stop own surviving group before waiting")
                return self.returncode

        def killpg(pid, sig):
            test.assertIn(pid, [p.pid for p in created], "must never signal unrelated processes")
            groups.append((pid, sig))
            process = next(p for p in created if p.pid == pid)
            if sig == signal.SIGKILL or not resist_term:
                process.returncode = -int(sig)

        def sleep(_):
            if interrupt:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            else:
                test.fail("a later failed rank must be detected without waiting for the first rank")

        argv = ["launch_online_dp.py", "--dp-size", "2", "--tp-size", "8",
                "--dp-size-local", "2", "--dp-rank-start", "0", "--dp-address", "192.0.2.1",
                "--dp-rpc-port", "12321", "--vllm-start-port", "7000"]
        with patch.object(sys, "argv", argv), patch.object(subprocess, "Popen", Process), \
             patch.object(os, "killpg", killpg, create=True), \
             patch.object(signal, "SIGKILL", 9, create=True), \
             patch.object(signal, "signal", lambda sig, handler: handlers.update({sig: handler})), \
             patch("time.sleep", sleep), self.assertRaises(SystemExit) as result:
            exec(compile(self.source, "embedded-launcher.py", "exec"), {"__name__": "__main__"})
        return result.exception.code, created, groups, polled

    def test_peer_failure_propagates_while_rank_zero_survives(self):
        code, processes, groups, polled = self.exercise(failure=7)
        self.assertEqual(code, 7)
        self.assertIn(31001, polled)
        self.assertEqual(processes[0].command[2], "0,1,2,3,4,5,6,7")
        self.assertEqual(processes[1].command[2], "8,9,10,11,12,13,14,15")
        self.assertEqual(processes[0].command[3:6], ["7000", "2", "0"])
        self.assertEqual(processes[1].command[3:6], ["7001", "2", "1"])
        self.assertIn((31000, signal.SIGTERM), groups)

    def test_sigterm_cleans_up_owned_groups_with_bounded_escalation(self):
        code, _, groups, _ = self.exercise(interrupt=True, resist_term=True)
        self.assertEqual(code, 128 + signal.SIGTERM)
        for pid in [31000, 31001]:
            self.assertLess(groups.index((pid, signal.SIGTERM)), groups.index((pid, 9)))

    def test_chinese_mirror_preserves_helper_and_commands(self):
        chinese = yaml.safe_load((ROOT / "models/zh/MiniMaxAI/MiniMax-M2.7.yaml").read_text(encoding="utf-8"))
        self.assertEqual(self.source, launcher_source(chinese))
        fences = lambda value: re.findall(r"```(?:bash|python)\n(.*?)\n```", value, re.S)
        for en, zh in zip(self.recipe["scenarios"], chinese["scenarios"], strict=True):
            self.assertNotIn("scripts", en, "TP>1 public commands must not imply converter support")
            self.assertEqual([fences(s["content"]) for s in en["steps"]],
                             [fences(s["content"]) for s in zh["steps"]])


if __name__ == "__main__":
    unittest.main()
