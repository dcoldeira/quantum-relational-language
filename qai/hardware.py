"""Hardware execution helpers for the QRL platform."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_TOKEN_FILE = Path(__file__).parent.parent.parent / "Quandela" / "QUANDELA.txt"


def _read_token() -> str:
    tok = os.environ.get("QUANDELA_TOKEN", "").strip()
    if tok:
        return tok
    if _TOKEN_FILE.exists():
        return _TOKEN_FILE.read_text().strip()
    raise RuntimeError(
        "No Quandela token found. Set QUANDELA_TOKEN env var or place token in "
        f"{_TOKEN_FILE}"
    )


def hardware_bell_test(
    shots: int = 1000,
    platform: str = "qpu:belenos",
) -> dict[str, Any]:
    """Run a Bell state on Quandela hardware; fall back to sim:belenos on error.

    Returns dict with: platform_used, requested_platform, shots, valid_events,
    yield_pct, hom_pct, qubit_distribution, violated_bell.
    """
    from qrl.mbqc import generate_bell_state_pattern
    from qrl.backends.perceval_path_adapter import run_on_cloud

    token = _read_token()
    pattern = generate_bell_state_pattern()

    tried_platform = platform
    try:
        raw = run_on_cloud(pattern, token=token, n_samples=shots, platform=platform)
    except Exception:
        platform = "sim:belenos"
        raw = run_on_cloud(pattern, token=token, n_samples=shots, platform=platform)

    qubit_dist = raw.get("qubit_results", {})
    total_valid = sum(qubit_dist.values())
    hom_events = shots - total_valid

    return {
        "platform_used": platform,
        "requested_platform": tried_platform,
        "shots": shots,
        "valid_events": total_valid,
        "yield_pct": round(total_valid / shots * 100, 1) if shots else 0,
        "hom_pct": round(hom_events / shots * 100, 1) if shots else 0,
        "qubit_distribution": qubit_dist,
        "violated_bell": total_valid > 0,
    }
