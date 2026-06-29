# imagehandler

Python CLI toolkit for image preprocessing.

Features:
- remove image backgrounds with pluggable backends
- split four-view character sheets without assuming equal-width columns
- extract equipment, item, and icon sprites from sheets
- judge processed results with PASS/WARN/FAIL quality reports
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

## Judge processed outputs

After image processing, run a quality gate:

```bash
imagehandler judge output.png --task remove-bg --debug
imagehandler judge out/ --task split-sheet --expected-count 4
imagehandler judge out/ --task extract-items --min-count 1
```

The judge command writes `*.judge.json` or `out/judge.json` with:

- `verdict`: `PASS`, `WARN`, or `FAIL`
- `score`: 0 to 100
- `failures`: hard failures that should block automation
- `warnings`: suspicious quality signals
- `metrics`: alpha, foreground, bbox, component, and size-consistency measurements

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
