"""Safe QRL code execution for the platform loop."""

from __future__ import annotations

import re
import traceback
from typing import Any

import math
import numpy as np


def _blocked_import(name, *args, **kwargs):
    raise ImportError(
        f"Imports are not allowed in the QRL sandbox. "
        f"All needed names (np, QuantumNetwork, chsh_test, etc.) are pre-loaded."
    )


def _build_namespace() -> dict:
    """Build the restricted execution namespace with QRL imports."""
    from qrl.domains.networks import (
        QuantumNetwork, ChannelSpec,
        fiber_channel, free_space_channel, ideal_channel, memory_noise,
    )
    from qrl.causal import (
        CPTPMap, ProcessMatrix, QuantumSwitch,
        QuantumCausalDAG, QuantumMarkovChain,
        depolarizing_channel, dephasing_channel, amplitude_damping_channel,
        cptp_from_unitary, vonneumann_entropy,
        quantum_mutual_information, quantum_conditional_mutual_information,
        is_quantum_conditionally_independent, petz_recovery_map,
    )
    from qrl.physics.bell import chsh_test, theoretical_chsh, BellTest
    from qrl.physics.ghz import mermin_test, GHZTest, ghz_paradox_test
    from qai.hardware import hardware_bell_test

    return {
        # stdlib
        "math": math,
        # numpy
        "np": np,
        # domain — include short aliases the model may generate
        "QuantumNetwork": QuantumNetwork,
        "Network": QuantumNetwork,          # alias: model sometimes emits Network
        "ChannelSpec": ChannelSpec,
        "fiber_channel": fiber_channel,
        "free_space_channel": free_space_channel,
        "ideal_channel": ideal_channel,
        "memory_noise": memory_noise,
        # causal
        "CPTPMap": CPTPMap,
        "ProcessMatrix": ProcessMatrix,
        "QuantumSwitch": QuantumSwitch,
        "QuantumCausalDAG": QuantumCausalDAG,
        "QuantumMarkovChain": QuantumMarkovChain,
        "depolarizing_channel": depolarizing_channel,
        "dephasing_channel": dephasing_channel,
        "amplitude_damping_channel": amplitude_damping_channel,
        "cptp_from_unitary": cptp_from_unitary,
        "vonneumann_entropy": vonneumann_entropy,
        "quantum_mutual_information": quantum_mutual_information,
        "quantum_conditional_mutual_information": quantum_conditional_mutual_information,
        "is_quantum_conditionally_independent": is_quantum_conditionally_independent,
        "petz_recovery_map": petz_recovery_map,
        # hardware
        "hardware_bell_test": hardware_bell_test,
        # physics
        "chsh_test": chsh_test,
        "theoretical_chsh": theoretical_chsh,
        "BellTest": BellTest,
        "mermin_test": mermin_test,
        "GHZTest": GHZTest,
        "ghz_paradox_test": ghz_paradox_test,
        # builtins (minimal safe set)
        "print": print,
        "round": round,
        "abs": abs,
        "min": min,
        "max": max,
        "len": len,
        "list": list,
        "dict": dict,
        "str": str,
        "float": float,
        "int": int,
        "bool": bool,
        "complex": complex,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        "range": range,
        "sum": sum,
        "zip": zip,
        "enumerate": enumerate,
        "sorted": sorted,
        "reversed": reversed,
        "any": any,
        "all": all,
        "chr": chr,
        "ord": ord,
        "isinstance": isinstance,
        "type": type,
        "callable": callable,
        "hasattr": hasattr,
        "getattr": getattr,
        "vars": vars,
        # result placeholder
        "result": None,
    }


def _extract_code(raw: str) -> str:
    """Strip markdown code fences if the LLM wrapped its output."""
    # Match ```python ... ``` or ``` ... ```
    fence = re.search(r"```(?:python)?\n?(.*?)```", raw, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return raw.strip()


def _strip_imports(code: str) -> str:
    """Remove import statements from LLM-generated code.

    All needed names are pre-loaded in the execution namespace, so import
    statements are both unnecessary and broken (no __import__ in sandbox).
    """
    lines = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        lines.append(line)
    return "\n".join(lines)


class ExecutionResult:
    def __init__(self, value: Any = None, error: str = "", code: str = "") -> None:
        self.value = value
        self.error = error
        self.code = code
        self.ok = not bool(error)

    def __repr__(self) -> str:
        if self.error:
            return f"ExecutionResult(error={self.error!r})"
        return f"ExecutionResult(value={self.value!r})"


def execute(raw_code: str) -> ExecutionResult:
    """Execute LLM-generated QRL code in a restricted namespace.

    The code must assign its final answer to a variable called `result`.

    Parameters
    ----------
    raw_code : Python code string (may include markdown fences)

    Returns
    -------
    ExecutionResult with .value (the `result` variable) or .error
    """
    code = _strip_imports(_extract_code(raw_code))
    ns = _build_namespace()
    try:
        exec(code, {"__builtins__": {"__import__": _blocked_import}}, ns)
    except Exception:
        return ExecutionResult(error=traceback.format_exc(), code=code)
    return ExecutionResult(value=ns.get("result"), code=code)
