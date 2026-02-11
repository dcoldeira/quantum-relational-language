"""
QRL Backend Adapters

Converts QRL MeasurementPatterns to various quantum computing frameworks.
"""

__all__ = []

try:
    from .graphix_adapter import qrl_to_graphix, GraphixConversionError, validate_conversion
    __all__ += ['qrl_to_graphix', 'GraphixConversionError', 'validate_conversion']
except ImportError:
    pass

try:
    from .perceval_adapter import qrl_to_perceval, PercevalConversionError
    __all__ += ['qrl_to_perceval', 'PercevalConversionError']
except ImportError:
    pass

try:
    from .perceval_path_adapter import (
        qrl_to_perceval_path,
        PathEncodingError,
        PathEncodedCircuit,
        interpret_path_results,
        run_on_cloud,
    )
    __all__ += [
        'qrl_to_perceval_path', 'PathEncodingError', 'PathEncodedCircuit',
        'interpret_path_results', 'run_on_cloud',
    ]
except ImportError:
    pass

from .pennylane_adapter import (
    qrl_to_pennylane,
    PennylaneConversionError,
    PennylaneCircuit,
    simulate_pennylane,
)
__all__ += [
    'qrl_to_pennylane', 'PennylaneConversionError',
    'PennylaneCircuit', 'simulate_pennylane',
]
