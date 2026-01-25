"""
QRL to Perceval Adapter

Provides the full pipeline: QRL → graphix → Perceval

Includes compatibility fixes for graphix API changes.
"""

import sys
from typing import Optional, Any

from ..mbqc.measurement_pattern import MeasurementPattern
from .graphix_adapter import qrl_to_graphix


class PercevalConversionError(Exception):
    """Raised when conversion to Perceval fails."""
    pass


def _patch_graphix_pattern(pattern):
    """
    Patch graphix Pattern to be compatible with graphix-perceval.

    graphix-perceval expects older API:
    - get_graph() -> now extract_graph()
    - get_measurement_commands() -> now need to extract from pattern
    """
    from graphix.command import M

    # Add get_graph as alias for extract_graph
    if not hasattr(pattern, 'get_graph'):
        pattern.get_graph = lambda: (
            list(pattern.extract_graph().nodes()),
            list(pattern.extract_graph().edges())
        )

    # Add get_measurement_commands
    if not hasattr(pattern, 'get_measurement_commands'):
        def get_measurement_commands():
            """Extract measurement commands in format expected by graphix-perceval."""
            commands = []
            for cmd in pattern:
                if isinstance(cmd, M):
                    # Format: (M, node, plane, angle, s_domain, t_domain)
                    commands.append((
                        'M',
                        cmd.node,
                        cmd.plane,
                        cmd.angle,
                        getattr(cmd, 's_domain', []),
                        getattr(cmd, 't_domain', [])
                    ))
            return commands
        pattern.get_measurement_commands = get_measurement_commands

    return pattern


def qrl_to_perceval(qrl_pattern: MeasurementPattern,
                    graphix_perceval_path: Optional[str] = None) -> Any:
    """
    Convert QRL MeasurementPattern to Perceval experiment.

    Pipeline: QRL → graphix → Perceval

    Args:
        qrl_pattern: QRL MeasurementPattern object
        graphix_perceval_path: Path to graphix-perceval if not installed

    Returns:
        PercevalExperiment object

    Raises:
        PercevalConversionError: If conversion fails
        ImportError: If dependencies not available
    """
    # Add graphix-perceval to path if specified
    if graphix_perceval_path:
        if graphix_perceval_path not in sys.path:
            sys.path.insert(0, graphix_perceval_path)

    # Try to import graphix-perceval
    try:
        from graphix_perceval import to_perceval
    except ImportError:
        raise ImportError(
            "graphix-perceval is required. Either install it or provide path:\n"
            "  qrl_to_perceval(pattern, graphix_perceval_path='/path/to/graphix-perceval')"
        )

    # Step 1: Convert QRL to graphix
    try:
        graphix_pattern = qrl_to_graphix(qrl_pattern)
    except Exception as e:
        raise PercevalConversionError(f"QRL → graphix conversion failed: {e}")

    # Step 2: Patch pattern for compatibility
    graphix_pattern = _patch_graphix_pattern(graphix_pattern)

    # Step 3: Convert graphix to Perceval
    try:
        perceval_exp = to_perceval(graphix_pattern)
    except Exception as e:
        raise PercevalConversionError(f"graphix → Perceval conversion failed: {e}")

    return perceval_exp


def test_perceval_available() -> bool:
    """Check if Perceval is available."""
    try:
        import perceval
        return True
    except ImportError:
        return False


def test_graphix_perceval_available(path: Optional[str] = None) -> bool:
    """Check if graphix-perceval is available."""
    if path:
        sys.path.insert(0, path)
    try:
        from graphix_perceval import to_perceval
        return True
    except ImportError:
        return False
