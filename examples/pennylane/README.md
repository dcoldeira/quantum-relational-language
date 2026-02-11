# QRL → PennyLane Examples

QRL compiles MBQC measurement patterns to PennyLane QNodes, enabling simulation
on `default.qubit` and execution on Xanadu hardware via PennyLane plugins.

## Setup

```bash
pip install pennylane
```

## Running

```bash
python examples/pennylane/bell_state.py
```

## How it works

QRL's MBQC compiler produces `MeasurementPattern` objects. The PennyLane adapter
converts these to PennyLane circuits:

| MBQC concept | PennyLane mapping |
|---|---|
| Preparation (&#124;+⟩) | `qml.Hadamard(wires=i)` |
| Entanglement (CZ) | `qml.CZ(wires=[i, j])` |
| Measurement | Basis rotation + `qml.measure()` |
| Correction | `qml.cond(m, qml.PauliX/Z)` |

This is QRL's second backend adapter (alongside Quandela/Perceval), demonstrating
cross-platform MBQC compilation.
