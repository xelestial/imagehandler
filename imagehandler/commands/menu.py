from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from .background import batch_remove_cmd, remove_cmd
from .items import batch_extract_cmd, extract_cmd
from .quality import batch_judge_cmd, judge_cmd
from .sheet import batch_split_cmd, split_cmd

app = typer.Typer(help="Interactive menu launcher.", invoke_without_command=True, no_args_is_help=False)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


DEFAULT_CONFIG = {
    "profile": "balanced",
    "bg": {
        "single": False,
        "recursive": True,
        "pattern": None,
        "backend": "auto",
        "model": None,
        "alpha_matting": False,
        "retry_on_fail": True,
        "continue_on_error": True,
    },
    "sheet": {
        "single": False,
        "recursive": True,
        "pattern": None,
        "views": 4,
        "padding": 24,
        "min_area": 1000,
        "merge_distance": 24,
        "normalize_size": None,
        "threshold": 28.0,
        "debug": True,
        "retry_on_fail": True,
        "continue_on_error": True,
        "accept_verdict": "PASS",
        "min_score": 85.0,
    },
    "items": {
        "single": False,
        "recursive": True,
        "pattern": None,
        "padding": 16,
        "min_area": 120,
        "merge_distance": 12,
        "square_canvas": False,
        "normalize_size": None,
        "transparent_bg": False,
        "threshold": 28.0,
        "debug": True,
        "retry_on_fail": True,
        "min_count": 1,
        "continue_on_error": True,
        "accept_verdict": "PASS",
        "min_score": 85.0,
    },
    "quality": {
        "single": False,
        "recursive": True,
        "pattern": None,
        "task": "auto",
        "alpha_required": None,
        "continue_on_error": True,
        "expected_count": None,
        "min_count": 1,
        "debug": False,
    },
}

PROFILES = {
    "fast": {
        "bg": {"retry_on_fail": False, "alpha_matting": False},
        "sheet": {"debug": False, "retry_on_fail": False},
        "items": {"debug": False, "retry_on_fail": False},
    },
    "balanced": {
        "bg": {"retry_on_fail": True, "alpha_matting": False},
        "sheet": {"debug": True, "retry_on_fail": True},
        "items": {"debug": True, "retry_on_fail": True},
    },
    "high_quality": {
        "bg": {"retry_on_fail": True, "alpha_matting": True},
        "sheet": {"debug": True, "retry_on_fail": True, "min_score": 90.0},
        "items": {"debug": True, "retry_on_fail": True, "min_score": 90.0},
    },
}


def _workspace_root() -> Path:
    return Path(os.environ.get("IMAGEHANDLER_WORKSPACE", "workspace"))


def _ensure_workspace() -> Path:
    root = _workspace_root()
    for rel in ["inbox/bg", "inbox/sheets", "inbox/items", "jobs", "archive", "_reports"]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


def _config_path() -> Path:
    return _workspace_root() / "config.json"


def _deep_update(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_config() -> dict:
    root = _ensure_workspace()
    path = root / "config.json"
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path.exists():
        try:
            _deep_update(config, json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            typer.echo(f"[WARN] Failed to read config: {path}. Using defaults.")
    return config


def _save_config(config: dict) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Saved config: {path}")


def _apply_profile(config: dict, profile: str) -> dict:
    if profile not in PROFILES:
        raise KeyError(profile)
    config["profile"] = profile
    _deep_update(config, PROFILES[profile])
    return config


def _show_workspace_hint() -> None:
    root = _ensure_workspace()
    typer.echo("\nWorkspace folders are ready:")
    typer.echo(f"  Background images      : {root / 'inbox' / 'bg'}")
    typer.echo(f"  Character sheets       : {root / 'inbox' / 'sheets'}")
    typer.echo(f"  Item/equipment sheets  : {root / 'inbox' / 'items'}")
    typer.echo(f"  Results                : {root / 'jobs'}")
    typer.echo("\nPut files into the matching inbox folder.")
    typer.echo("Quick run uses these folders automatically.")


def _choose(title: str, options: list[tuple[str, str]]) -> str:
    typer.echo(f"\n{title}")
    for idx, (_, label) in enumerate(options, start=1):
        typer.echo(f"  {idx}. {label}")
    while True:
        raw = typer.prompt("Select number")
        try:
            index = int(raw)
        except ValueError:
            typer.echo("Please enter a number.")
            continue
        if 1 <= index <= len(options):
            return options[index - 1][0]
        typer.echo("Selection out of range.")


def _default_input_for(group: str) -> Path:
    root = _ensure_workspace()
    if group == "bg":
        return root / "inbox" / "bg"
    if group == "sheet":
        return root / "inbox" / "sheets"
    if group == "items":
        return root / "inbox" / "items"
    return root


def _list_images(path: Path, recursive: bool = True) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    pattern = "**/*" if recursive else "*"
    return sorted([p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _warn_empty_folder(path: Path) -> bool:
    files = _list_images(path, recursive=True)
    if not files:
        typer.echo(f"\n[WARN] No image files found in: {path}")
        typer.echo("Add image files to that folder, then run Quick Run again.\n")
        return True
    typer.echo(f"Found {len(files)} image file(s) in: {path}")
    return False


def _prompt_path(label: str, default: Path | None = None, must_exist: bool = False, directory_ok: bool = True) -> Path:
    while True:
        if default is not None:
            value = typer.prompt(f"{label} [default: {default}]", default="").strip()
            p = default if not value else Path(value)
        else:
            value = typer.prompt(label).strip()
            p = Path(value)
        if must_exist and not p.exists():
            typer.echo(f"Path does not exist: {p}")
            continue
        if must_exist and not directory_ok and p.is_dir():
            typer.echo("A file path is required. Try again.")
            continue
        return p


def _prompt_optional_path(label: str) -> Path | None:
    value = typer.prompt(f"{label} (leave blank for auto job/workspace output)", default="").strip()
    return Path(value) if value else None


def _prompt_optional_text(label: str) -> str | None:
    value = typer.prompt(f"{label} (leave blank for auto)", default="").strip()
    return value or None


def _show_current_config(config: dict) -> None:
    typer.echo("\nCurrent quick-run config")
    typer.echo(f"  Profile: {config.get('profile', 'balanced')}")
    typer.echo(f"  BG     : single={config['bg']['single']} backend={config['bg']['backend']} retry={config['bg']['retry_on_fail']}")
    typer.echo(f"  Sheet  : single={config['sheet']['single']} views={config['sheet']['views']} retry={config['sheet']['retry_on_fail']}")
    typer.echo(f"  Items  : single={config['items']['single']} retry={config['items']['retry_on_fail']} min_count={config['items']['min_count']}")
    typer.echo(f"  Judge  : single={config['quality']['single']} task={config['quality']['task']}")


def _config_menu(config: dict) -> dict:
    while True:
        _show_current_config(config)
        choice = _choose(
            "Config menu",
            [
                ("profile", "Choose profile (fast / balanced / high_quality)"),
                ("mode", "Set quick-run mode (single or batch) per task"),
                ("reset", "Reset config to defaults"),
                ("back", "Back to main menu"),
            ],
        )
        if choice == "profile":
            profile = _choose(
                "Choose profile",
                [
                    ("fast", "Fast - fewer retries, minimal debug"),
                    ("balanced", "Balanced - recommended default"),
                    ("high_quality", "High quality - more retries / stricter quality"),
                ],
            )
            config = _apply_profile(config, profile)
            _save_config(config)
            continue
        if choice == "mode":
            task = _choose(
                "Choose task",
                [
                    ("bg", "Background removal"),
                    ("sheet", "Character sheet splitting"),
                    ("items", "Item/equipment extraction"),
                    ("quality", "Quality judge"),
                ],
            )
            mode = _choose("Choose quick-run mode", [("batch", "Batch folder"), ("single", "Single file")])
            config[task]["single"] = mode == "single"
            _save_config(config)
            continue
        if choice == "reset":
            config = json.loads(json.dumps(DEFAULT_CONFIG))
            _save_config(config)
            continue
        return config


def _quick_run_bg(config: dict) -> None:
    opts = config["bg"]
    default = _default_input_for("bg")
    if opts.get("single"):
        path = _prompt_path("Input image path", default=default, must_exist=True)
        if path.is_dir():
            files = _list_images(path, recursive=True)
            if not files:
                _warn_empty_folder(path)
                return
            typer.echo(f"Using first image in folder: {files[0]}")
            path = files[0]
        remove_cmd(path, None, _workspace_root(), None, opts["backend"], opts.get("model"), opts["alpha_matting"], False, False, 0.0, opts["retry_on_fail"], "PASS", 85.0)
        return
    folder = default
    if _warn_empty_folder(folder):
        return
    batch_remove_cmd(folder, None, _workspace_root(), opts["recursive"], opts.get("pattern"), opts["backend"], opts.get("model"), opts["alpha_matting"], opts["retry_on_fail"], opts["continue_on_error"])


def _quick_run_sheet(config: dict) -> None:
    opts = config["sheet"]
    default = _default_input_for("sheet")
    if opts.get("single"):
        path = _prompt_path("Input sheet image", default=default, must_exist=True)
        if path.is_dir():
            files = _list_images(path, recursive=True)
            if not files:
                _warn_empty_folder(path)
                return
            typer.echo(f"Using first image in folder: {files[0]}")
            path = files[0]
        split_cmd(path, None, _workspace_root(), None, opts["views"], opts["padding"], opts["min_area"], opts["merge_distance"], opts.get("normalize_size"), opts["threshold"], opts["debug"], opts["retry_on_fail"], opts["accept_verdict"], opts["min_score"])
        return
    folder = default
    if _warn_empty_folder(folder):
        return
    batch_split_cmd(folder, None, _workspace_root(), opts["recursive"], opts.get("pattern"), opts["views"], opts["padding"], opts["min_area"], opts["merge_distance"], opts.get("normalize_size"), opts["threshold"], opts["debug"], opts["retry_on_fail"], opts["continue_on_error"])


def _quick_run_items(config: dict) -> None:
    opts = config["items"]
    default = _default_input_for("items")
    if opts.get("single"):
        path = _prompt_path("Input item sheet", default=default, must_exist=True)
        if path.is_dir():
            files = _list_images(path, recursive=True)
            if not files:
                _warn_empty_folder(path)
                return
            typer.echo(f"Using first image in folder: {files[0]}")
            path = files[0]
        extract_cmd(path, None, _workspace_root(), None, opts["padding"], opts["min_area"], opts["merge_distance"], opts["square_canvas"], opts.get("normalize_size"), opts["transparent_bg"], opts["threshold"], opts["debug"], opts["retry_on_fail"], opts["accept_verdict"], opts["min_score"], opts["min_count"])
        return
    folder = default
    if _warn_empty_folder(folder):
        return
    batch_extract_cmd(folder, None, _workspace_root(), opts["recursive"], opts.get("pattern"), opts["padding"], opts["min_area"], opts["merge_distance"], opts["square_canvas"], opts.get("normalize_size"), opts["transparent_bg"], opts["threshold"], opts["debug"], opts["retry_on_fail"], opts["min_count"], opts["continue_on_error"])


def _quick_run_quality(config: dict) -> None:
    opts = config["quality"]
    default = _workspace_root() / "jobs"
    if opts.get("single"):
        path = _prompt_path("Target image or directory", default=default, must_exist=True)
        judge_cmd(path, opts["task"], opts.get("expected_count"), opts["min_count"], None, _workspace_root(), None, opts["debug"], opts.get("alpha_required"))
        return
    if not default.exists() or not any(default.iterdir()):
        typer.echo(f"\n[WARN] No job outputs found in: {default}")
        typer.echo("Run bg/sheet/items first, then judge outputs.\n")
        return
    batch_judge_cmd(default, None, _workspace_root(), True, opts.get("pattern"), opts["task"], opts.get("alpha_required"), opts["continue_on_error"])


def _quick_run_menu(config: dict) -> None:
    _show_current_config(config)
    choice = _choose(
        "Quick run",
        [
            ("bg", "Background removal"),
            ("sheet", "Character sheet splitting"),
            ("items", "Item/equipment extraction"),
            ("quality", "Quality judge"),
            ("back", "Back"),
        ],
    )
    if choice == "bg":
        _quick_run_bg(config)
    elif choice == "sheet":
        _quick_run_sheet(config)
    elif choice == "items":
        _quick_run_items(config)
    elif choice == "quality":
        _quick_run_quality(config)


def _advanced_menu() -> None:
    group = _choose(
        "Choose a menu group",
        [
            ("bg", "Background removal"),
            ("sheet", "Character sheet splitting"),
            ("items", "Item/equipment extraction"),
            ("quality", "Quality judge"),
        ],
    )

    if group == "bg":
        action = _choose("Choose an action", [("remove", "Remove background (single file)"), ("batch", "Remove background (batch folder)")])
        if action == "remove":
            default = _default_input_for("bg")
            input_path = _prompt_path("Input image path", default=default, must_exist=True)
            if input_path.is_dir():
                _warn_empty_folder(input_path)
                typer.echo("Single-file mode needs one image file. Use batch mode for folders.")
                return
            output = _prompt_optional_path("Output PNG path")
            workspace = _workspace_root()
            job = _prompt_optional_text("Job name")
            backend = typer.prompt("Backend", default="auto")
            model = _prompt_optional_text("Model name")
            alpha_matting = typer.confirm("Enable alpha matting?", default=False)
            mask_only = typer.confirm("Write mask only?", default=False)
            no_postprocess = typer.confirm("Disable postprocess cleanup?", default=False)
            feather = float(typer.prompt("Feather blur radius", default="0.0"))
            retry_on_fail = typer.confirm("Retry on quality failure?", default=True)
            accept_verdict = typer.prompt("Accept verdict", default="PASS")
            min_score = float(typer.prompt("Minimum score", default="85"))
            remove_cmd(input_path, output, workspace, job, backend, model, alpha_matting, mask_only, no_postprocess, feather, retry_on_fail, accept_verdict, min_score)
            return
        input_path = _prompt_path("Input folder", default=_default_input_for("bg"), must_exist=True)
        _warn_empty_folder(input_path)
        output_dir = _prompt_optional_path("Output folder")
        workspace = _workspace_root()
        recursive = typer.confirm("Search recursively?", default=True)
        pattern = _prompt_optional_text("Glob pattern")
        backend = typer.prompt("Backend", default="auto")
        model = _prompt_optional_text("Model name")
        alpha_matting = typer.confirm("Enable alpha matting?", default=False)
        retry_on_fail = typer.confirm("Retry on quality failure?", default=True)
        continue_on_error = typer.confirm("Continue on error?", default=True)
        batch_remove_cmd(input_path, output_dir, workspace, recursive, pattern, backend, model, alpha_matting, retry_on_fail, continue_on_error)
        return

    if group == "sheet":
        action = _choose("Choose an action", [("split", "Split sheet (single file)"), ("batch", "Split sheet (batch folder)")])
        if action == "split":
            default = _default_input_for("sheet")
            input_path = _prompt_path("Input sheet image", default=default, must_exist=True)
            if input_path.is_dir():
                _warn_empty_folder(input_path)
                typer.echo("Single-file mode needs one sheet image. Use batch mode for folders.")
                return
            output_dir = _prompt_optional_path("Output folder")
            workspace = _workspace_root()
            job = _prompt_optional_text("Job name")
            views = int(typer.prompt("Expected number of views", default="4"))
            padding = int(typer.prompt("Padding", default="24"))
            min_area = int(typer.prompt("Minimum area", default="1000"))
            merge_distance = int(typer.prompt("Merge distance", default="24"))
            normalize_size_text = typer.prompt("Normalize size (blank for none)", default="").strip()
            normalize_size = int(normalize_size_text) if normalize_size_text else None
            threshold = float(typer.prompt("Threshold", default="28.0"))
            debug = typer.confirm("Write debug images?", default=True)
            retry_on_fail = typer.confirm("Retry on quality failure?", default=True)
            accept_verdict = typer.prompt("Accept verdict", default="PASS")
            min_score = float(typer.prompt("Minimum score", default="85"))
            split_cmd(input_path, output_dir, workspace, job, views, padding, min_area, merge_distance, normalize_size, threshold, debug, retry_on_fail, accept_verdict, min_score)
            return
        input_path = _prompt_path("Input sheet folder", default=_default_input_for("sheet"), must_exist=True)
        _warn_empty_folder(input_path)
        output_dir = _prompt_optional_path("Output folder")
        workspace = _workspace_root()
        recursive = typer.confirm("Search recursively?", default=True)
        pattern = _prompt_optional_text("Glob pattern")
        views = int(typer.prompt("Expected number of views", default="4"))
        padding = int(typer.prompt("Padding", default="24"))
        min_area = int(typer.prompt("Minimum area", default="1000"))
        merge_distance = int(typer.prompt("Merge distance", default="24"))
        normalize_size_text = typer.prompt("Normalize size (blank for none)", default="").strip()
        normalize_size = int(normalize_size_text) if normalize_size_text else None
        threshold = float(typer.prompt("Threshold", default="28.0"))
        debug = typer.confirm("Write debug images?", default=True)
        retry_on_fail = typer.confirm("Retry on quality failure?", default=True)
        continue_on_error = typer.confirm("Continue on error?", default=True)
        batch_split_cmd(input_path, output_dir, workspace, recursive, pattern, views, padding, min_area, merge_distance, normalize_size, threshold, debug, retry_on_fail, continue_on_error)
        return

    if group == "items":
        action = _choose("Choose an action", [("extract", "Extract items (single file)"), ("batch", "Extract items (batch folder)")])
        if action == "extract":
            default = _default_input_for("items")
            input_path = _prompt_path("Input equipment/item sheet", default=default, must_exist=True)
            if input_path.is_dir():
                _warn_empty_folder(input_path)
                typer.echo("Single-file mode needs one item sheet. Use batch mode for folders.")
                return
            output_dir = _prompt_optional_path("Output folder")
            workspace = _workspace_root()
            job = _prompt_optional_text("Job name")
            padding = int(typer.prompt("Padding", default="16"))
            min_area = int(typer.prompt("Minimum area", default="120"))
            merge_distance = int(typer.prompt("Merge distance", default="12"))
            square_canvas = typer.confirm("Square canvas outputs?", default=False)
            normalize_size_text = typer.prompt("Normalize size (blank for none)", default="").strip()
            normalize_size = int(normalize_size_text) if normalize_size_text else None
            transparent_bg = typer.confirm("Transparent background outputs?", default=False)
            threshold = float(typer.prompt("Threshold", default="28.0"))
            debug = typer.confirm("Write debug images?", default=True)
            retry_on_fail = typer.confirm("Retry on quality failure?", default=True)
            accept_verdict = typer.prompt("Accept verdict", default="PASS")
            min_score = float(typer.prompt("Minimum score", default="85"))
            min_count = int(typer.prompt("Minimum extracted item count", default="1"))
            extract_cmd(input_path, output_dir, workspace, job, padding, min_area, merge_distance, square_canvas, normalize_size, transparent_bg, threshold, debug, retry_on_fail, accept_verdict, min_score, min_count)
            return
        input_path = _prompt_path("Input item sheet folder", default=_default_input_for("items"), must_exist=True)
        _warn_empty_folder(input_path)
        output_dir = _prompt_optional_path("Output folder")
        workspace = _workspace_root()
        recursive = typer.confirm("Search recursively?", default=True)
        pattern = _prompt_optional_text("Glob pattern")
        padding = int(typer.prompt("Padding", default="16"))
        min_area = int(typer.prompt("Minimum area", default="120"))
        merge_distance = int(typer.prompt("Merge distance", default="12"))
        square_canvas = typer.confirm("Square canvas outputs?", default=False)
        normalize_size_text = typer.prompt("Normalize size (blank for none)", default="").strip()
        normalize_size = int(normalize_size_text) if normalize_size_text else None
        transparent_bg = typer.confirm("Transparent background outputs?", default=False)
        threshold = float(typer.prompt("Threshold", default="28.0"))
        debug = typer.confirm("Write debug images?", default=True)
        retry_on_fail = typer.confirm("Retry on quality failure?", default=True)
        min_count = int(typer.prompt("Minimum extracted item count", default="1"))
        continue_on_error = typer.confirm("Continue on error?", default=True)
        batch_extract_cmd(input_path, output_dir, workspace, recursive, pattern, padding, min_area, merge_distance, square_canvas, normalize_size, transparent_bg, threshold, debug, retry_on_fail, min_count, continue_on_error)
        return

    action = _choose("Choose an action", [("judge", "Judge one processed file"), ("batch", "Judge many processed files")])
    if action == "judge":
        target = _prompt_path("Target image or directory", must_exist=True)
        task = typer.prompt("Task", default="auto")
        expected_count_text = typer.prompt("Expected count (blank for none)", default="").strip()
        expected_count = int(expected_count_text) if expected_count_text else None
        min_count = int(typer.prompt("Minimum count", default="1"))
        output = _prompt_optional_path("Judge JSON output path")
        workspace = _workspace_root()
        job = _prompt_optional_text("Job name")
        debug = typer.confirm("Write debug judge mask?", default=False)
        alpha_required_text = typer.prompt("Require alpha? (true/false/blank)", default="").strip().lower()
        alpha_required = None if not alpha_required_text else alpha_required_text in {"1", "true", "y", "yes"}
        judge_cmd(target, task, expected_count, min_count, output, workspace, job, debug, alpha_required)
        return
    input_path = _prompt_path("Input file or folder to judge", must_exist=True)
    output_dir = _prompt_optional_path("Output folder for judge JSON files")
    workspace = _workspace_root()
    recursive = typer.confirm("Search recursively?", default=True)
    pattern = _prompt_optional_text("Glob pattern")
    task = typer.prompt("Task", default="auto")
    alpha_required_text = typer.prompt("Require alpha? (true/false/blank)", default="").strip().lower()
    alpha_required = None if not alpha_required_text else alpha_required_text in {"1", "true", "y", "yes"}
    continue_on_error = typer.confirm("Continue on error?", default=True)
    batch_judge_cmd(input_path, output_dir, workspace, recursive, pattern, task, alpha_required, continue_on_error)


@app.callback()
def menu_callback() -> None:
    _show_workspace_hint()
    config = _load_config()
    while True:
        choice = _choose(
            "Main menu",
            [
                ("quick", "Quick run (use saved config, minimal prompts)"),
                ("config", "Config"),
                ("advanced", "Advanced menu"),
                ("exit", "Exit"),
            ],
        )
        if choice == "quick":
            _quick_run_menu(config)
            continue
        if choice == "config":
            config = _config_menu(config)
            continue
        if choice == "advanced":
            _advanced_menu()
            continue
        typer.echo("Bye.")
        return
