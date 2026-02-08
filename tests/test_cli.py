"""Tests for the QRL CLI."""

import io
import os
import sys
import unittest
from unittest.mock import patch

from qrl.cli import build_parser, main, QRLShell


class TestParser(unittest.TestCase):
    """Test the argparse configuration."""

    def setUp(self):
        self.parser = build_parser()

    def test_version_flag(self):
        args = self.parser.parse_args(["--version"])
        self.assertTrue(args.version)

    def test_run_bell_defaults(self):
        args = self.parser.parse_args(["run", "bell"])
        self.assertEqual(args.command, "run")
        self.assertEqual(args.experiment, "bell")
        self.assertEqual(args.shots, 1000)
        self.assertFalse(args.verbose)

    def test_run_bell_options(self):
        args = self.parser.parse_args(["run", "bell", "--shots", "500", "-v"])
        self.assertEqual(args.shots, 500)
        self.assertTrue(args.verbose)

    def test_run_ghz_defaults(self):
        args = self.parser.parse_args(["run", "ghz"])
        self.assertEqual(args.experiment, "ghz")
        self.assertEqual(args.shots, 1000)
        self.assertEqual(args.qubits, 3)

    def test_run_ghz_options(self):
        args = self.parser.parse_args(["run", "ghz", "--qubits", "4", "--shots", "200"])
        self.assertEqual(args.qubits, 4)
        self.assertEqual(args.shots, 200)

    def test_run_demo(self):
        args = self.parser.parse_args(["run", "demo", "--quick", "--section", "2"])
        self.assertEqual(args.experiment, "demo")
        self.assertTrue(args.quick)
        self.assertEqual(args.section, 2)

    def test_compile(self):
        args = self.parser.parse_args(["compile", "bell", "--target", "graphix"])
        self.assertEqual(args.command, "compile")
        self.assertEqual(args.experiment, "bell")
        self.assertEqual(args.target, "graphix")

    def test_compile_default_target(self):
        args = self.parser.parse_args(["compile", "ghz"])
        self.assertEqual(args.target, "perceval")

    def test_inspect(self):
        args = self.parser.parse_args(["inspect", "graph", "bell"])
        self.assertEqual(args.command, "inspect")
        self.assertEqual(args.artifact, "graph")
        self.assertEqual(args.experiment, "bell")

    def test_cloud_status(self):
        args = self.parser.parse_args(["cloud", "status"])
        self.assertEqual(args.command, "cloud")
        self.assertEqual(args.cloud_command, "status")

    def test_cloud_run(self):
        args = self.parser.parse_args(["cloud", "run", "bell", "--platform", "sim:belenos"])
        self.assertEqual(args.cloud_command, "run")
        self.assertEqual(args.experiment, "bell")
        self.assertEqual(args.platform, "sim:belenos")

    def test_info(self):
        args = self.parser.parse_args(["info"])
        self.assertEqual(args.command, "info")

    def test_shell(self):
        args = self.parser.parse_args(["shell"])
        self.assertEqual(args.command, "shell")


class TestMainDispatch(unittest.TestCase):
    """Test that main() dispatches to the right handlers."""

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_version(self, mock_out):
        main(["--version"])
        self.assertIn("0.2.0", mock_out.getvalue())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_no_args_prints_help(self, mock_out):
        main([])
        output = mock_out.getvalue()
        self.assertIn("qrl", output.lower())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_info(self, mock_out):
        main(["info"])
        output = mock_out.getvalue()
        self.assertIn("QRL", output)
        self.assertIn("numpy", output.lower())


class TestRunBell(unittest.TestCase):
    """Test qrl run bell."""

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_bell(self, mock_out):
        main(["run", "bell", "--shots", "100"])
        output = mock_out.getvalue()
        self.assertIn("S =", output)
        self.assertIn("bell", output.lower())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_bell_verbose(self, mock_out):
        main(["run", "bell", "--shots", "100", "-v"])
        output = mock_out.getvalue()
        self.assertIn("S =", output)
        self.assertIn("Correlation", output)


class TestRunGHZ(unittest.TestCase):
    """Test qrl run ghz."""

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_ghz(self, mock_out):
        main(["run", "ghz", "--shots", "100"])
        output = mock_out.getvalue()
        self.assertIn("M =", output)
        self.assertIn("Mermin", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_ghz_verbose(self, mock_out):
        main(["run", "ghz", "--shots", "100", "-v"])
        output = mock_out.getvalue()
        self.assertIn("M =", output)


class TestInspect(unittest.TestCase):
    """Test qrl inspect commands."""

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_inspect_graph_bell(self, mock_out):
        main(["inspect", "graph", "bell"])
        output = mock_out.getvalue()
        self.assertIn("Nodes", output)
        self.assertIn("Edges", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_inspect_graph_ghz(self, mock_out):
        main(["inspect", "graph", "ghz"])
        output = mock_out.getvalue()
        self.assertIn("Nodes", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_inspect_pattern_bell(self, mock_out):
        main(["inspect", "pattern", "bell"])
        output = mock_out.getvalue()
        self.assertIn("Preparation", output)
        self.assertIn("Entanglement", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_inspect_pattern_ghz(self, mock_out):
        main(["inspect", "pattern", "ghz"])
        output = mock_out.getvalue()
        self.assertIn("pattern", output.lower())


class TestCloudStatus(unittest.TestCase):
    """Test qrl cloud status."""

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cloud_status(self, mock_out):
        main(["cloud", "status"])
        output = mock_out.getvalue()
        self.assertIn("perceval", output.lower())
        self.assertIn("sim:belenos", output)


class TestQRLShell(unittest.TestCase):
    """Test the interactive REPL."""

    def _run_shell(self, commands: str) -> str:
        """Feed commands to the REPL and return stdout."""
        shell = QRLShell()
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        sys.stdin = io.StringIO(commands)
        sys.stdout = buf = io.StringIO()
        try:
            shell.cmdloop()
        except SystemExit:
            pass
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout
        return buf.getvalue()

    def test_entangle_and_list(self):
        output = self._run_shell("entangle test 2\nlist\nquit\n")
        self.assertIn("Created 'test'", output)
        self.assertIn("test", output)

    def test_entangle_and_graph(self):
        output = self._run_shell("entangle foo 3 --type ghz\ngraph foo\nquit\n")
        self.assertIn("Created 'foo'", output)
        self.assertIn("Nodes", output)

    def test_compile_relation(self):
        output = self._run_shell("entangle bar 2\ncompile bar\nquit\n")
        self.assertIn("Pattern for 'bar'", output)
        self.assertIn("Preparation", output)

    def test_chsh(self):
        output = self._run_shell("chsh --shots 100\nquit\n")
        self.assertIn("S =", output)

    def test_mermin(self):
        output = self._run_shell("mermin --shots 100\nquit\n")
        self.assertIn("M =", output)

    def test_info(self):
        output = self._run_shell("info\nquit\n")
        self.assertIn("QRL", output)

    def test_empty_list(self):
        output = self._run_shell("list\nquit\n")
        self.assertIn("No relations", output)

    def test_unknown_relation(self):
        output = self._run_shell("graph nonexistent\nquit\n")
        self.assertIn("Unknown relation", output)


if __name__ == "__main__":
    unittest.main()
