# Quandela Cloud Examples

Examples demonstrating the QRL → Quandela Cloud pipeline.

## Setup

1. Get API token from [cloud.quandela.com](https://cloud.quandela.com)
2. Set environment variable or enter when prompted:
   ```bash
   export QUANDELA_TOKEN="your-token-here"
   ```

## Files

| File | Description |
|------|-------------|
| `cloud_test.py` | Basic connection test (beamsplitter) |
| `bell_state.py` | Full QRL Bell state → cloud pipeline |
| `perceval_hello_world.py` | Perceval basics (local simulation) |

## Running

```bash
# Test cloud connection
python examples/quandela/cloud_test.py

# Run QRL Bell state on cloud
python examples/quandela/bell_state.py
```

## Pipeline

```
QRL MeasurementPattern
    ↓ qrl_to_graphix()
graphix Pattern
    ↓ to_perceval()
Perceval Circuit
    ↓ RemoteProcessor
Quandela Cloud (sim:belenos)
    ↓
Photonic sampling results
```
