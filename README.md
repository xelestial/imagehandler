# imagehandler

`imagehandler` is a Python CLI toolkit for production-style image preprocessing:

- remove image backgrounds with pluggable backends (`rembg`, `transparent-background`, classical CV fallback)
- split 4-view character sheets without assuming equal-width columns
- extract equipment/items/icons from a sheet using whitespace and foreground components
- save debug overlays and JSON reports so failed automatic decisions can be inspected

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Background removal backends are optional:

```bash
pip install -e ".[bg]"          # rembg CPU
pip install -e ".[bg-gpu]"      # rembg GPU / onnxruntime-gpu path
pip install -e ".[transparent]" # InSPyReNet-based transparent-background backend
pip install -e ".[matting]"     # optional alpha-matting helpers
```


## Recommended workspace layout

Each job gets its own folder under `workspace/jobs/`:

```text
workspace/
  inbox/
    bg/
    sheets/
    items/
  jobs/
    sample_job/
      input/
        bg/
        sheets/
        items/
      output/
        bg/
        sheets/
        items/
        quality/
        logs/
      tmp/
      README.txt
  archive/
  _reports/
```

You can choose a different workspace root or initial job folder name:

```bash
./setup.sh --workspace ./workspace --job-name project_001
```

During actual CLI work, you do not have to name a job manually. If you omit `--job` and also omit `--output`, ImageHandler creates a per-file job folder automatically using the input filename. If that job folder name already exists, it appends a datetime suffix such as `character_sheet_20260629_131023`.


## Default inbox workflow

`./run.sh` creates these folders before launching the menu:

```text
workspace/
  inbox/
    bg/       # put background-removal source images here
    sheets/   # put character-sheet source images here
    items/    # put item/equipment-sheet source images here
  jobs/       # per-job outputs
```

Run:

```bash
./run.sh
```

The menu shows the inbox folder paths. When it asks for an input path, press Enter to use the matching default folder.

## Interactive menu selection

If you prefer choosing interactively, launch the menu selector:

```bash
imagehandler menu
```

It shows numbered menu items, then asks for the required paths and options before executing the selected command.

## Menu-style CLI groups

The CLI is now organized by menu item groups. Old commands remain as aliases, but the preferred form is grouped:

```bash
imagehandler bg remove input.png -o output.png
imagehandler bg remove input.png --workspace ./workspace
imagehandler bg batch-remove ./inputs --workspace ./workspace --recursive --retry-on-fail

imagehandler sheet split character_sheet.png -o out --views 4
imagehandler sheet split character_sheet.png --workspace ./workspace --views 4
imagehandler sheet batch-split ./sheets --workspace ./workspace --recursive --retry-on-fail

imagehandler items extract equipment_sheet.png -o out --min-count 4 --retry-on-fail
imagehandler items extract equipment_sheet.png --workspace ./workspace --retry-on-fail
imagehandler items batch-extract ./equipment_sheets --workspace ./workspace --recursive

imagehandler quality judge output.png --task remove-bg
imagehandler quality judge output.png --workspace ./workspace --task remove-bg
imagehandler quality batch-judge ./outputs --workspace ./workspace --recursive
```

Feature folders added in the package:

```text
imagehandler/commands/   # menu item command groups
imagehandler/batch/      # multi-file discovery and output mapping helpers
```

Legacy aliases still work:

```bash
imagehandler remove-bg input.png -o output.png
imagehandler split-sheet sheet.png -o out
imagehandler extract-items sheet.png -o out
imagehandler judge output.png
```

## Commands

### Remove background

```bash
imagehandler remove-bg input.png -o output.png --backend auto
imagehandler remove-bg input.png -o output.png --backend rembg --model isnet-anime
imagehandler remove-bg input.png -o output.png --backend rembg --model birefnet-general --alpha-matting
imagehandler remove-bg input.png -o output.png --backend transparent
imagehandler remove-bg input.png -o output.png --backend classical
```

Output sidecars:

```text
output.png
output.mask.png
output.report.json
```

### Split a 4-view character sheet

```bash
imagehandler split-sheet character_sheet.png -o out --views 4 --debug
```

This does **not** blindly cut `image_width / 4`. It detects foreground regions, groups nearby fragments, and only falls back to projection/equal splitting when needed.

### Extract equipment/items/icons from a sheet

```bash
imagehandler extract-items equipment_sheet.png -o out --padding 24 --transparent-bg --debug
imagehandler extract-items equipment_sheet.png -o out --square-canvas --normalize-size 512
```

Output:

```text
item_001.png
item_002.png
...
manifest.json
debug_mask.png
debug_boxes.png
```


## Automatic retry / fallback

If a result fails quality checks, the tool can automatically try another backend or alternate strategies.

```bash
imagehandler remove-bg input.png -o output.png --retry-on-fail
imagehandler remove-bg input.png -o output.png --backend rembg --retry-on-fail --accept-verdict WARN --min-score 75
imagehandler split-sheet character_sheet.png -o out --retry-on-fail
imagehandler extract-items equipment_sheet.png -o out --retry-on-fail --min-count 4
```

Behavior:

- `remove-bg`: tries another library/backend in order (`rembg` -> `transparent-background` -> classical fallback)
- `split-sheet`: retries alternate detection parameter strategies
- `extract-items`: retries alternate threshold / min-area / merge-distance strategies
- writes `*.fallback.json` or `out/fallback.json` summarizing all attempts and the selected result

## Design principle

Automatic image editing must leave evidence. Every command returns a report containing boxes, confidence-like metrics, warnings, and the algorithm path that was used.

## macOS setup

```bash
chmod +x setup.sh
./setup.sh
```

The setup script now also creates a recommended workspace and one initial per-job folder so outputs do not get mixed together.

For Apple Silicon or Intel Mac, use the default CPU backend. The `--gpu` option is for NVIDIA/CUDA systems and will automatically fall back to CPU on normal macOS.

```bash
./setup.sh --transparent --matting
./setup.sh --all
```

If Python 3.11-3.13 is not found on macOS, install a supported Python version first, for example:

```bash
brew install python@3.12
PYTHON_BIN=python3.12 ./setup.sh
```

## Judge processed outputs

After processing images, run a quality gate to decide whether the result can be used automatically or needs manual review.

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

Typical checks include:

- alpha/transparency exists after background removal
- foreground is not almost empty or almost full-canvas
- object does not touch the canvas border unexpectedly
- connected component count is plausible
- split-sheet output count matches the expected four views
- split output sizes are reasonably consistent
- extracted items are not near-empty crops


## Auto job naming

When `--output` is omitted, these commands automatically create a job folder:

- `imagehandler bg remove ... --workspace ./workspace`
- `imagehandler sheet split ... --workspace ./workspace`
- `imagehandler items extract ... --workspace ./workspace`
- `imagehandler quality judge ... --workspace ./workspace`

Rules:

- if `--job` is given, that folder name is used
- if `--job` is omitted, the input filename stem is used
- if the derived job folder already exists, a datetime suffix is appended
- the source file is copied into the job `input/...` folder for record keeping when possible

Example:

```text
input file: /data/hero_sheet.png
first run job: workspace/jobs/hero_sheet/
second run job: workspace/jobs/hero_sheet_20260629_131023/
```

## Quick run + Config workflow

The menu now starts with:

- **Quick run**: minimal prompts, uses saved config and default inbox folders automatically
- **Config**: choose a profile and single/batch mode once
- **Advanced menu**: full manual control

Recommended flow:

1. Put files into `workspace/inbox/bg`, `workspace/inbox/sheets`, or `workspace/inbox/items`
2. Run `./run.sh`
3. Choose `Quick run`
4. If needed, choose `Config` once and save your preferred profile

The config file is stored at `workspace/config.json`.

## Batch completion handling

When you run a batch task from an input folder, successfully processed source images are moved into a `complete/` folder under that same input root.

Example:

```text
workspace/inbox/sheets/
  character_a.png
  character_b.png

# after batch split
workspace/inbox/sheets/
  complete/
    character_a.png
    character_b.png
```

This prevents the same files from being processed again on the next batch run. Files already inside `complete/` are automatically ignored.
