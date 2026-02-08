"""
QRL Command-Line Interface

Usage:
    qrl run bell [--shots N] [--verbose]
    qrl run ghz [--shots N] [--qubits N] [--verbose]
    qrl run demo [--quick] [--section N]
    qrl compile bell|ghz [--target T] [-o FILE]
    qrl inspect graph|pattern|circuit bell|ghz
    qrl cloud status
    qrl cloud run bell|ghz [--platform P]
    qrl info
    qrl shell
"""

import argparse
import cmd
import os
import sys
import textwrap
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Color helpers (respect NO_COLOR and non-tty)
# ---------------------------------------------------------------------------

def _use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _use_color() else text


def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m" if _use_color() else text


def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m" if _use_color() else text


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m" if _use_color() else text


def _cyan(text: str) -> str:
    return f"\033[36m{text}\033[0m" if _use_color() else text


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _check_dep(name: str) -> bool:
    """Return True if *name* is importable."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _require_dep(name: str, install_hint: str = "") -> Any:
    """Import and return *name*, or exit with a helpful message."""
    try:
        return __import__(name)
    except ImportError:
        hint = f"  pip install {install_hint}" if install_hint else ""
        print(_red(f"Error: '{name}' is not installed."))
        if hint:
            print(hint)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def _get_bell_pattern():
    from qrl.mbqc import generate_bell_state_pattern
    return generate_bell_state_pattern()


def _get_ghz_pattern(n_qubits: int = 3):
    from qrl.mbqc import generate_ghz_state_pattern
    return generate_ghz_state_pattern(n_qubits)


def _get_pattern_for(experiment: str, n_qubits: int = 3):
    if experiment == "bell":
        return _get_bell_pattern()
    elif experiment == "ghz":
        return _get_ghz_pattern(n_qubits)
    else:
        print(_red(f"Unknown experiment: {experiment}"))
        sys.exit(1)


# ===================================================================
# Command handlers
# ===================================================================

def cmd_run_bell(args: argparse.Namespace) -> None:
    from qrl.physics.bell import BellTest

    test = BellTest()
    shots = args.shots

    if args.verbose:
        print(_bold("Theoretical predictions"))
        print(test.predict())
        print()

    print(f"Running Bell/CHSH test ({shots} trials per setting)...")
    result = test.run(trials=shots, verbose=args.verbose)
    print()

    status = _green("VIOLATED") if result.violated else _yellow("not violated")
    print(_bold("Result"))
    print(f"  S = {result.S:.4f}  (classical limit 2.0, quantum max ~2.828)")
    print(f"  Classical bound: {status}")
    print(f"  Violation: {result.violation_amount:+.4f}")
    print(f"  Fraction of maximum: {result.fraction_of_maximum:.1%}")

    if args.verbose:
        print()
        print(_bold("Correlations"))
        for label, val in result.correlations.items():
            print(f"  {label} = {val:+.4f}")

    if args.verbose:
        print()
        print(_bold("Theory vs Observation"))
        print(test.compare(trials=shots))


def cmd_run_ghz(args: argparse.Namespace) -> None:
    from qrl.physics.ghz import GHZTest

    n = getattr(args, "qubits", 3) or 3
    test = GHZTest(n_qubits=n)
    shots = args.shots

    if args.verbose:
        print(_bold("Theoretical predictions"))
        print(test.predict())
        print()

    print(f"Running GHZ/Mermin test ({n} qubits, {shots} trials)...")
    mermin = test.run_mermin(trials=shots, verbose=args.verbose)
    print()

    status = _green("VIOLATED") if mermin.violated else _yellow("not violated")
    print(_bold("Mermin inequality"))
    print(f"  M = {mermin.M:.4f}  (classical limit 2.0)")
    print(f"  Classical bound: {status}")
    print(f"  Violation ratio: {mermin.violation_ratio:.2f}x")

    if args.verbose:
        print()
        print(_bold("Correlations"))
        for label, val in mermin.correlations.items():
            print(f"  {label} = {val:+.4f}")

    print()
    print("Running GHZ paradox test...")
    paradox = test.run_paradox(trials=min(shots, 20), verbose=args.verbose)
    status = _green("YES") if paradox.paradox_demonstrated else _red("NO")
    print(f"  Paradox demonstrated: {status}")

    if args.verbose:
        print()
        print(_bold("Theory vs Observation"))
        print(test.compare(trials=shots))


def cmd_run_demo(args: argparse.Namespace) -> None:
    from qrl.physics.demo import run_demo

    run_demo(
        quick=args.quick,
        section=args.section,
        interactive=not args.quick,
    )


def cmd_compile(args: argparse.Namespace) -> None:
    target = args.target
    experiment = args.experiment
    pattern = _get_pattern_for(experiment)

    print(f"Compiling {experiment} to {target}...")
    print()

    if target == "perceval":
        try:
            from qrl.backends import qrl_to_perceval_path
            result = qrl_to_perceval_path(pattern)
            print(_bold("Perceval path-encoded circuit"))
            print(f"  Logical qubits: {result.n_qubits}")
            print(f"  Optical modes:  {result.n_modes}")
            print(f"  Input state:    {result.input_state}")

            if args.output:
                # Write a summary to file
                with open(args.output, "w") as f:
                    f.write(f"# QRL compile {experiment} -> perceval\n")
                    f.write(f"qubits: {result.n_qubits}\n")
                    f.write(f"modes:  {result.n_modes}\n")
                    f.write(f"input:  {result.input_state}\n")
                print(f"\n  Written to {args.output}")
        except ImportError:
            print(_red("perceval-quandela is not installed."))
            print("  pip install perceval-quandela")
            sys.exit(1)

    elif target == "graphix":
        try:
            from qrl.backends import qrl_to_graphix
            gx_pattern = qrl_to_graphix(pattern)
            print(_bold("Graphix pattern"))
            print(f"  {gx_pattern}")
        except ImportError:
            print(_red("graphix is not installed."))
            print("  pip install graphix")
            sys.exit(1)

    elif target == "qiskit":
        try:
            from qrl.compiler import compile_to_qiskit
            circuit = compile_to_qiskit(pattern)
            print(_bold("Qiskit circuit"))
            print(circuit)
        except ImportError:
            print(_red("qiskit is not installed."))
            print("  pip install qiskit")
            sys.exit(1)
        except Exception as exc:
            print(_red(f"Compilation error: {exc}"))
            sys.exit(1)

    else:
        print(_red(f"Unknown target: {target}"))
        sys.exit(1)


def cmd_inspect(args: argparse.Namespace) -> None:
    artifact = args.artifact
    experiment = args.experiment
    pattern = _get_pattern_for(experiment)

    if artifact == "graph":
        from qrl.core import QRLProgram
        from qrl.mbqc import extract_graph

        program = QRLProgram(f"{experiment} inspection")
        if experiment == "bell":
            q = [program.create_system() for _ in range(2)]
            relation = program.entangle(*q)
        else:
            n = getattr(args, "qubits", 3) or 3
            q = [program.create_system() for _ in range(n)]
            relation = program.entangle(*q, state_type="ghz")

        graph = extract_graph(relation)
        print(_bold(f"Graph state ({experiment})"))
        print(f"  Nodes: {list(graph.nodes())}")
        print(f"  Edges: {list(graph.edges())}")
        print()
        print(_bold("Adjacency"))
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            print(f"  {node} -> {neighbors}")

    elif artifact == "pattern":
        print(_bold(f"Measurement pattern ({experiment})"))
        print(f"  Description:  {pattern.description}")
        print(f"  Preparation:  {pattern.preparation}")
        print(f"  Entanglement: {pattern.entanglement}")
        print(f"  Output:       {pattern.output_qubits}")
        print(f"  Num qubits:   {pattern.num_qubits}")
        if pattern.measurements:
            print()
            print(_bold("Measurements"))
            for m in pattern.measurements:
                print(f"  {m}")
        if pattern.corrections:
            print()
            print(_bold("Corrections"))
            for c in pattern.corrections:
                print(f"  {c}")

    elif artifact == "circuit":
        try:
            from qrl.backends import qrl_to_perceval_path
            result = qrl_to_perceval_path(pattern)
            import perceval as pcvl
            print(_bold(f"Perceval circuit ({experiment})"))
            print(f"  Qubits: {result.n_qubits}  Modes: {result.n_modes}")
            print()
            print(pcvl.pdisplay(result.circuit, compact=True))
        except ImportError:
            print(_red("perceval-quandela is not installed (needed for circuit view)."))
            print("  pip install perceval-quandela")
            sys.exit(1)
    else:
        print(_red(f"Unknown artifact: {artifact}"))
        sys.exit(1)


def cmd_cloud_status(args: argparse.Namespace) -> None:
    print(_bold("Cloud platform availability"))
    print()

    deps = [
        ("perceval-quandela", "perceval", "pip install perceval-quandela"),
        ("graphix", "graphix", "pip install graphix"),
        ("qiskit", "qiskit", "pip install qiskit"),
    ]
    for label, mod, hint in deps:
        ok = _check_dep(mod)
        mark = _green("installed") if ok else _yellow("not installed")
        print(f"  {label:25s} {mark}")

    print()

    token = os.environ.get("QUANDELA_TOKEN") or os.environ.get("QRL_CLOUD_TOKEN")
    if token:
        print(f"  Quandela token:          {_green('set')}")
    else:
        print(f"  Quandela token:          {_yellow('not set')}")
        print("    Export QUANDELA_TOKEN to enable cloud execution")

    print()
    print(f"  Default platform:        sim:belenos (simulator)")


def cmd_cloud_run(args: argparse.Namespace) -> None:
    token = os.environ.get("QUANDELA_TOKEN") or os.environ.get("QRL_CLOUD_TOKEN")
    if not token:
        print(_red("No Quandela token found."))
        print("  Export QUANDELA_TOKEN=<your-token>")
        sys.exit(1)

    experiment = args.experiment
    platform = args.platform
    pattern = _get_pattern_for(experiment)

    print(f"Running {experiment} on {platform}...")

    try:
        from qrl.backends import run_on_cloud
        results = run_on_cloud(
            pattern,
            token=token,
            n_samples=args.shots,
            platform=platform,
        )
        print()
        print(_bold("Cloud results"))
        print(f"  Platform:  {results.get('platform', platform)}")
        print(f"  Samples:   {results.get('n_samples', args.shots)}")
        print(f"  Qubits:    {results.get('n_qubits', '?')}")
        print(f"  Modes:     {results.get('n_modes', '?')}")
        if "qubit_results" in results:
            print()
            print(_bold("Qubit-level results"))
            for state, count in sorted(
                results["qubit_results"].items(), key=lambda x: -x[1]
            )[:10]:
                print(f"  {state}: {count}")
    except ImportError:
        print(_red("perceval-quandela is not installed."))
        print("  pip install perceval-quandela")
        sys.exit(1)
    except Exception as exc:
        print(_red(f"Cloud execution error: {exc}"))
        sys.exit(1)


def cmd_help(args: argparse.Namespace) -> None:
    import qrl

    print(_bold(f"QRL {qrl.__version__}") + " - A relations-first quantum programming language")
    print()
    print(_bold("USAGE"))
    print("  qrl <command> [options]")
    print()
    print(_bold("EXPERIMENTS"))
    print(f"  {_cyan('qrl run bell')} [--shots N] [--verbose]         Run Bell/CHSH inequality test")
    print(f"  {_cyan('qrl run ghz')}  [--shots N] [--qubits N] [-v]  Run GHZ/Mermin inequality test")
    print(f"  {_cyan('qrl run demo')} [--quick] [--section N]         Run the interactive physics demo")
    print()
    print(_bold("COMPILATION"))
    print(f"  {_cyan('qrl compile')} bell|ghz [--target T] [-o FILE]  Compile to backend")
    print( "      Targets: perceval (default), graphix, qiskit")
    print(f"  {_cyan('qrl inspect')} graph|pattern|circuit bell|ghz   Inspect compilation artifacts")
    print()
    print(_bold("CLOUD"))
    print(f"  {_cyan('qrl cloud status')}                             Show platform availability")
    print(f"  {_cyan('qrl cloud run')} bell|ghz [--platform P]        Run on Quandela cloud")
    print( "      Requires QUANDELA_TOKEN env var. Default platform: sim:belenos")
    print()
    print(_bold("TOOLS"))
    print(f"  {_cyan('qrl info')}                                     Version, dependencies, source stats")
    print(f"  {_cyan('qrl shell')}                                    Interactive REPL")
    print(f"  {_cyan('qrl help')}                                     Show this help")
    print(f"  {_cyan('qrl --version')}                                Print version")
    print()
    print(_bold("EXAMPLES"))
    print(f"  $ qrl run bell --shots 500 -v")
    print(f"    Run CHSH test with 500 trials, show correlations")
    print()
    print(f"  $ qrl run ghz --qubits 4")
    print(f"    Run 4-qubit Mermin inequality + GHZ paradox")
    print()
    print(f"  $ qrl inspect pattern ghz")
    print(f"    Show MBQC measurement pattern for GHZ state")
    print()
    print(f"  $ qrl compile bell --target perceval -o bell.txt")
    print(f"    Compile Bell state to Perceval, write summary to file")
    print()
    print(f"  $ qrl shell")
    print(f"    Start interactive REPL, then:")
    print(f"      qrl> entangle mybell 2")
    print(f"      qrl> graph mybell")
    print(f"      qrl> compile mybell")
    print(f"      qrl> chsh --shots 500")
    print(f"      qrl> quit")
    print()
    print(_bold("SHELL COMMANDS"))
    print(f"  entangle <name> <n> [--type bell|ghz|w]  Create entangled relation")
    print(f"  ask <relation> <basis> [--subsystem N]    Measure a relation")
    print(f"  graph <relation>                          Show graph state topology")
    print(f"  compile <relation>                        Generate MBQC pattern")
    print(f"  chsh [--shots N]                          Run CHSH/Bell test")
    print(f"  mermin [--shots N]                        Run Mermin/GHZ test")
    print(f"  target perceval|graphix <relation>        Compile pattern to backend")
    print(f"  list                                      Show all relations/patterns")
    print(f"  info                                      Version and session info")
    print(f"  help                                      List commands")
    print(f"  quit                                      Exit the shell")


def cmd_info(args: argparse.Namespace) -> None:
    import qrl

    print(_bold(f"QRL {qrl.__version__}"))
    print("A relations-first quantum programming language")
    print()

    print(_bold("Dependencies"))
    deps = [
        ("numpy", "numpy"),
        ("networkx", "networkx"),
        ("qiskit", "qiskit"),
        ("cirq", "cirq"),
        ("perceval-quandela", "perceval"),
        ("graphix", "graphix"),
        ("matplotlib", "matplotlib"),
    ]
    for label, mod in deps:
        if _check_dep(mod):
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  {label:25s} {_green(ver)}")
        else:
            print(f"  {label:25s} {_yellow('not installed')}")

    print()
    print(_bold("Source statistics"))
    _print_source_stats()


def _print_source_stats() -> None:
    """Count lines in src/qrl/**/*.py relative to the package location."""
    try:
        import qrl
        pkg_dir = os.path.dirname(os.path.abspath(qrl.__file__))
    except Exception:
        print("  (could not locate source)")
        return

    total_lines = 0
    total_files = 0
    for root, _dirs, files in os.walk(pkg_dir):
        for fname in files:
            if fname.endswith(".py"):
                total_files += 1
                try:
                    with open(os.path.join(root, fname)) as f:
                        total_lines += sum(1 for _ in f)
                except OSError:
                    pass

    print(f"  Source files: {total_files}")
    print(f"  Total lines: {total_lines:,}")

    # Count tests if found
    tests_dir = os.path.join(os.path.dirname(pkg_dir), "..", "tests")
    tests_dir = os.path.normpath(tests_dir)
    if os.path.isdir(tests_dir):
        test_files = [f for f in os.listdir(tests_dir) if f.startswith("test_") and f.endswith(".py")]
        print(f"  Test files:   {len(test_files)}")


# ===================================================================
# Interactive REPL
# ===================================================================

class QRLShell(cmd.Cmd):
    intro = textwrap.dedent("""\
        QRL Interactive Shell
        Type 'help' for available commands, 'quit' to exit.
    """)
    prompt = "qrl> "

    def __init__(self):
        super().__init__()
        from qrl.core import QRLProgram
        self.program = QRLProgram("interactive")
        self.relations: Dict[str, Any] = {}
        self.patterns: Dict[str, Any] = {}

    # -- relation management --

    def do_entangle(self, line: str) -> None:
        """entangle <name> <n_qubits> [--type bell|ghz|w]
        Create a named entangled relation."""
        parts = line.split()
        if len(parts) < 2:
            print("Usage: entangle <name> <n_qubits> [--type bell|ghz|w]")
            return

        name = parts[0]
        try:
            n = int(parts[1])
        except ValueError:
            print("n_qubits must be an integer")
            return

        state_type = "ghz"
        if "--type" in parts:
            idx = parts.index("--type")
            if idx + 1 < len(parts):
                state_type = parts[idx + 1]

        if n == 2 and state_type == "ghz":
            state_type = "bell"

        systems = [self.program.create_system() for _ in range(n)]
        relation = self.program.entangle(*systems, state_type=state_type)
        self.relations[name] = relation
        print(f"Created '{name}': {n}-qubit {state_type} relation")
        print(f"  Entropy: {relation.entanglement_entropy:.4f}")

    def complete_entangle(self, text, line, begidx, endidx):
        if "--type" in line:
            return [t for t in ["bell", "ghz", "w"] if t.startswith(text)]
        return []

    def do_ask(self, line: str) -> None:
        """ask <relation> <basis> [--subsystem N]
        Measure a relation. basis: spin_z, spin_x, spin_y"""
        parts = line.split()
        if len(parts) < 2:
            print("Usage: ask <relation> <basis> [--subsystem N]")
            return

        name = parts[0]
        if name not in self.relations:
            print(f"Unknown relation '{name}'. Use 'list' to see available.")
            return

        basis = parts[1]
        from qrl.core import create_question, QuestionType

        type_map = {
            "spin_z": QuestionType.SPIN_Z,
            "spin_x": QuestionType.SPIN_X,
            "spin_y": QuestionType.SPIN_Y,
        }
        if basis not in type_map:
            print(f"Unknown basis '{basis}'. Use: spin_z, spin_x, spin_y")
            return

        subsystem = None
        if "--subsystem" in parts:
            idx = parts.index("--subsystem")
            if idx + 1 < len(parts):
                subsystem = int(parts[idx + 1])

        question = create_question(type_map[basis], subsystem=subsystem)
        result = self.program.ask(self.relations[name], question)
        print(f"  Outcome: {result}")

    def complete_ask(self, text, line, begidx, endidx):
        parts = line.split()
        if len(parts) <= 2:
            return [n for n in self.relations if n.startswith(text)]
        return [b for b in ["spin_z", "spin_x", "spin_y"] if b.startswith(text)]

    def do_graph(self, line: str) -> None:
        """graph <relation>
        Show graph state for a relation."""
        name = line.strip()
        if not name:
            print("Usage: graph <relation>")
            return
        if name not in self.relations:
            print(f"Unknown relation '{name}'.")
            return

        from qrl.mbqc import extract_graph
        graph = extract_graph(self.relations[name])
        print(f"Graph for '{name}':")
        print(f"  Nodes: {list(graph.nodes())}")
        print(f"  Edges: {list(graph.edges())}")

    def complete_graph(self, text, line, begidx, endidx):
        return [n for n in self.relations if n.startswith(text)]

    def do_compile(self, line: str) -> None:
        """compile <relation>
        Generate MBQC measurement pattern for a relation."""
        name = line.strip()
        if not name:
            print("Usage: compile <relation>")
            return
        if name not in self.relations:
            print(f"Unknown relation '{name}'.")
            return

        from qrl.mbqc import generate_pattern_from_relation
        pattern = generate_pattern_from_relation(self.relations[name])
        self.patterns[name] = pattern
        print(f"Pattern for '{name}':")
        print(f"  Description:  {pattern.description}")
        print(f"  Preparation:  {pattern.preparation}")
        print(f"  Entanglement: {pattern.entanglement}")
        print(f"  Output:       {pattern.output_qubits}")

    def complete_compile(self, text, line, begidx, endidx):
        return [n for n in self.relations if n.startswith(text)]

    def do_chsh(self, line: str) -> None:
        """chsh [--shots N]
        Run CHSH/Bell test."""
        shots = 1000
        parts = line.split()
        if "--shots" in parts:
            idx = parts.index("--shots")
            if idx + 1 < len(parts):
                shots = int(parts[idx + 1])

        from qrl.physics.bell import BellTest
        test = BellTest()
        result = test.run(trials=shots)
        status = "VIOLATED" if result.violated else "not violated"
        print(f"S = {result.S:.4f}  (limit 2.0)  {status}")

    def do_mermin(self, line: str) -> None:
        """mermin [--shots N]
        Run Mermin/GHZ test."""
        shots = 1000
        parts = line.split()
        if "--shots" in parts:
            idx = parts.index("--shots")
            if idx + 1 < len(parts):
                shots = int(parts[idx + 1])

        from qrl.physics.ghz import GHZTest
        test = GHZTest()
        result = test.run_mermin(trials=shots)
        status = "VIOLATED" if result.violated else "not violated"
        print(f"M = {result.M:.4f}  (limit 2.0)  {status}")

    def do_target(self, line: str) -> None:
        """target perceval|graphix <relation>
        Compile a relation's pattern to a backend target."""
        parts = line.split()
        if len(parts) < 2:
            print("Usage: target perceval|graphix <relation>")
            return

        backend = parts[0]
        name = parts[1]

        if name not in self.patterns:
            # Auto-compile if relation exists
            if name in self.relations:
                from qrl.mbqc import generate_pattern_from_relation
                self.patterns[name] = generate_pattern_from_relation(self.relations[name])
            else:
                print(f"No pattern for '{name}'. Run 'compile {name}' first.")
                return

        pattern = self.patterns[name]

        if backend == "perceval":
            try:
                from qrl.backends import qrl_to_perceval_path
                result = qrl_to_perceval_path(pattern)
                print(f"Perceval circuit for '{name}':")
                print(f"  Qubits: {result.n_qubits}  Modes: {result.n_modes}")
            except ImportError:
                print("perceval-quandela is not installed.")
        elif backend == "graphix":
            try:
                from qrl.backends import qrl_to_graphix
                gx = qrl_to_graphix(pattern)
                print(f"Graphix pattern for '{name}': {gx}")
            except ImportError:
                print("graphix is not installed.")
        else:
            print(f"Unknown backend '{backend}'. Use: perceval, graphix")

    def complete_target(self, text, line, begidx, endidx):
        parts = line.split()
        if len(parts) <= 2:
            return [b for b in ["perceval", "graphix"] if b.startswith(text)]
        return [n for n in list(self.relations) + list(self.patterns) if n.startswith(text)]

    def do_list(self, line: str) -> None:
        """List all named relations and patterns."""
        if not self.relations and not self.patterns:
            print("No relations or patterns created yet.")
            return
        if self.relations:
            print("Relations:")
            for name, rel in self.relations.items():
                print(f"  {name}: {len(rel.systems)}-qubit, entropy={rel.entanglement_entropy:.4f}")
        if self.patterns:
            print("Patterns:")
            for name, pat in self.patterns.items():
                print(f"  {name}: {pat.description}")

    def do_info(self, line: str) -> None:
        """Show QRL version and environment info."""
        import qrl
        print(f"QRL {qrl.__version__}")
        print(f"Relations: {len(self.relations)}  Patterns: {len(self.patterns)}")

    def do_quit(self, line: str) -> bool:
        """Exit the QRL shell."""
        return True

    do_exit = do_quit
    do_EOF = do_quit

    def emptyline(self) -> None:
        pass


# ===================================================================
# Argument parser
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qrl",
        description="QRL - A relations-first quantum programming language",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Show version and exit",
    )

    sub = parser.add_subparsers(dest="command")

    # -- run --
    run_parser = sub.add_parser("run", help="Run experiments")
    run_sub = run_parser.add_subparsers(dest="experiment")

    bell_p = run_sub.add_parser("bell", help="Run Bell/CHSH test")
    bell_p.add_argument("--shots", type=int, default=1000, help="Trials per setting (default: 1000)")
    bell_p.add_argument("--verbose", "-v", action="store_true")

    ghz_p = run_sub.add_parser("ghz", help="Run GHZ/Mermin test")
    ghz_p.add_argument("--shots", type=int, default=1000, help="Trials per setting (default: 1000)")
    ghz_p.add_argument("--qubits", type=int, default=3, help="Number of qubits (default: 3)")
    ghz_p.add_argument("--verbose", "-v", action="store_true")

    demo_p = run_sub.add_parser("demo", help="Run interactive demo")
    demo_p.add_argument("--quick", "-q", action="store_true", help="Quick mode")
    demo_p.add_argument("--section", "-s", type=int, choices=[1, 2, 3, 4, 5])

    # -- compile --
    compile_p = sub.add_parser("compile", help="Compile to backend")
    compile_p.add_argument("experiment", choices=["bell", "ghz"])
    compile_p.add_argument("--target", "-t", default="perceval",
                           choices=["perceval", "graphix", "qiskit"])
    compile_p.add_argument("--output", "-o", help="Output file")

    # -- inspect --
    inspect_p = sub.add_parser("inspect", help="Inspect compilation artifacts")
    inspect_p.add_argument("artifact", choices=["graph", "pattern", "circuit"])
    inspect_p.add_argument("experiment", choices=["bell", "ghz"])

    # -- cloud --
    cloud_p = sub.add_parser("cloud", help="Cloud execution")
    cloud_sub = cloud_p.add_subparsers(dest="cloud_command")

    cloud_sub.add_parser("status", help="Show platform availability")

    cloud_run_p = cloud_sub.add_parser("run", help="Run on Quandela cloud")
    cloud_run_p.add_argument("experiment", choices=["bell", "ghz"])
    cloud_run_p.add_argument("--platform", default="sim:belenos")
    cloud_run_p.add_argument("--shots", type=int, default=1000)

    # -- info --
    sub.add_parser("info", help="Show version, dependencies, and stats")

    # -- shell --
    sub.add_parser("shell", help="Interactive REPL")

    # -- help --
    sub.add_parser("help", help="Show full usage guide")

    return parser


# ===================================================================
# Main entry point
# ===================================================================

def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        import qrl
        print(f"qrl {qrl.__version__}")
        return

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        if args.experiment is None:
            print("Usage: qrl run {bell,ghz,demo}")
            sys.exit(1)
        if args.experiment == "bell":
            cmd_run_bell(args)
        elif args.experiment == "ghz":
            cmd_run_ghz(args)
        elif args.experiment == "demo":
            cmd_run_demo(args)

    elif args.command == "compile":
        cmd_compile(args)

    elif args.command == "inspect":
        cmd_inspect(args)

    elif args.command == "cloud":
        if args.cloud_command is None:
            print("Usage: qrl cloud {status,run}")
            sys.exit(1)
        if args.cloud_command == "status":
            cmd_cloud_status(args)
        elif args.cloud_command == "run":
            cmd_cloud_run(args)

    elif args.command == "info":
        cmd_info(args)

    elif args.command == "shell":
        QRLShell().cmdloop()

    elif args.command == "help":
        cmd_help(args)


if __name__ == "__main__":
    main()
