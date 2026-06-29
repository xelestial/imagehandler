# imagehandler

Python CLI toolkit for image preprocessing.

Features:
- remove image backgrounds with pluggable backends
- split four-view character sheets without assuming equal-width columns
- extract equipment, item, and icon sprites from sheets
- write debug overlays and JSON reports

## Setup

```bash
chmod +x setup.sh
./setup.sh
```

Optional:

```bash
./setup.sh --gpu
./setup.sh --transparent --matting
./setup.sh --all
```

## macOS

Use the default CPU backend on macOS:

```bash
chmod +x setup.sh
./setup.sh
```

If multiple Python versions exist or the system Python is too old, run:

```bash
brew install python@3.12
PYTHON_BIN=python3.12 ./setup.sh
```

`--gpu` is for NVIDIA/CUDA systems. On normal macOS, `setup.sh --gpu` automatically falls back to CPU.
