"""
QRL Backend Adapters

Converts QRL MeasurementPatterns to various quantum computing frameworks.
"""

from .graphix_adapter import qrl_to_graphix, GraphixConversionError, validate_conversion
from .perceval_adapter import qrl_to_perceval, PercevalConversionError
from .perceval_path_adapter import (
    qrl_to_perceval_path,
    PathEncodingError,
    PathEncodedCircuit,
    interpret_path_results,
    run_on_cloud,
)

__all__ = [
    'qrl_to_graphix',
    'GraphixConversionError',
    'validate_conversion',
    'qrl_to_perceval',
    'PercevalConversionError',
    # Path-encoded (cloud-compatible)
    'qrl_to_perceval_path',
    'PathEncodingError',
    'PathEncodedCircuit',
    'interpret_path_results',
    'run_on_cloud',
]
