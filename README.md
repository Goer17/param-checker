# Param Checker

Visualize PyTorch checkpoint parameters in a browser heatmap. The UI supports key autocomplete, zoom/pan, and hover values for large tensors via a tile-based API.

## Requirements

- Python 3.10+
- `uv` (for environment management)

## Setup (uv)

```bash
# Create virtual environment
uv venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Run with a checkpoint

```bash
python param-checker.py -i /path/to/model.pth
```

Open `http://127.0.0.1:8000` in your browser.

## Generate test checkpoints

Small test checkpoint:

```bash
python scripts/generate_test_checkpoint.py
python param-checker.py -i sample.pth
```

Large test checkpoint:

```bash
python scripts/generate_large_checkpoint.py
python param-checker.py -i large.pth
```

## Notes

- Tensor keys are flattened using dotted notation (for example `state_dict.layer1.weight`).
- Non-2D tensors are reshaped to 2D for visualization.
- Large tensors use `/api/meta`, `/api/tile`, and `/api/value` to avoid sending full matrices to the browser.
