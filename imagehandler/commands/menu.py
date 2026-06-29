from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from .background import batch_remove_cmd
from .items import batch_extract_cmd
from .sheet import batch_split_cmd

app = typer.Typer(help="Interactive menu launcher.", invoke_without_command=True, no_args_is_help=False)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_CONFIG = {
    "profile": "balanced",
    "bg": {"backend": "auto", "model": None, "alpha_matting": False, "retry_on_fail": True, "continue_on_error": True, "recursive": True},
    "sheet": {"views": 4, "padding": 24, "min_area": 1000, "merge_distance": 24, "normalize_size": None, "threshold": 28.0, "debug": True, "retry_on_fail": True, "continue_on_error": True},
    "items": {"padding": 16, "min_area": 120, "merge_distance": 12, "square_canvas": False, "normalize_size": None, "transparent_bg": False, "threshold": 28.0, "debug": True, "retry_on_fail": True, "min_count": 1, "continue_on_error": True},
}

PROFILES = {
    "fast": {"bg": {"retry_on_fail": False}, "sheet": {"debug": False, "retry_on_fail": False}, "items": {"debug": False, "retry_on_fail": False}},
    "balanced": {"bg": {"retry_on_fail": True}, "sheet": {"debug": True, "retry_on_fail": True}, "items": {"debug": True, "retry_on_fail": True}},
    "high_quality": {"bg": {"retry_on_fail": True, "alpha_matting": True}, "sheet": {"debug": True, "retry_on_fail": True}, "items": {"debug": True, "retry_on_fail": True}},
}


def _workspace_root() -> Path:
    return Path(os.environ.get("IMAGEHANDLER_WORKSPACE", "workspace"))


def _ensure_workspace() -> Path:
    root = _workspace_root()
    for task in ["bg", "sheets", "items"]:
        for rel in ["input", "jobs", "failed"]:
            (root / task / rel).mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    return root


def _deep_update(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _config_path() -> Path:
    return _ensure_workspace() / "config.json"


def _load_config() -> dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    path = _config_path()
    if path.exists():
        try:
            _deep_update(config, json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            typer.echo(f"[WARN] Failed to read config: {path}. Using defaults.")
    return config


def _save_config(config: dict) -> None:
    _config_path().write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Saved config: {_config_path()}")


def _apply_profile(config: dict, profile: str) -> dict:
    config["profile"] = profile
    _deep_update(config, PROFILES[profile])
    return config


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


def _task_paths(task: str) -> tuple[Path, Path, Path]:
    root = _ensure_workspace()
    task_dir = {"bg": "bg", "sheet": "sheets", "items": "items"}[task]
    task_root = root / task_dir
    return task_root / "input", task_root / "jobs", task_root / "failed"


def _list_images(path: Path, recursive: bool = True) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    pattern = "**/*" if recursive else "*"
    ignored = {"jobs", "failed", "reports", "tmp"}
    return sorted([p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not any(part in ignored for part in p.relative_to(path).parts)])


def _show_hint() -> None:
    root = _ensure_workspace()
    typer.echo("\nOptimized workspace. Put source files into the task input folder:")
    typer.echo(f"  BG     : {root / 'bg' / 'input'}")
    typer.echo(f"  Sheets : {root / 'sheets' / 'input'}")
    typer.echo(f"  Items  : {root / 'items' / 'input'}")
    typer.echo("\nSuccess flow:")
    typer.echo(f"  {root}/<task>/input/source.png")
    typer.echo(f"  -> {root}/<task>/jobs/<job>/input/source.png")
    typer.echo(f"  -> {root}/<task>/jobs/<job>/output/")
    typer.echo("\nFailure flow:")
    typer.echo(f"  {root}/<task>/failed/source.png")


def _show_config(config: dict) -> None:
    typer.echo("\nCurrent config")
    typer.echo(f"  Profile: {config.get('profile')}")
    typer.echo(f"  BG retry    : {config['bg']['retry_on_fail']}")
    typer.echo(f"  Sheet retry : {config['sheet']['retry_on_fail']} / views={config['sheet']['views']}")
    typer.echo(f"  Items retry : {config['items']['retry_on_fail']}")


def _config_menu(config: dict) -> dict:
    while True:
        _show_config(config)
        choice = _choose("Config", [("profile", "Choose profile"), ("reset", "Reset defaults"), ("back", "Back")])
        if choice == "profile":
            profile = _choose("Profile", [("fast", "Fast"), ("balanced", "Balanced"), ("high_quality", "High quality")])
            _apply_profile(config, profile)
            _save_config(config)
        elif choice == "reset":
            config = json.loads(json.dumps(DEFAULT_CONFIG))
            _save_config(config)
        else:
            return config


def _show_task_help(task: str, title: str) -> tuple[Path, list[Path]]:
    input_dir, jobs_dir, failed_dir = _task_paths(task)
    pending = _list_images(input_dir, recursive=True)
    typer.echo(f"\n{title}")
    typer.echo(f"  Put source files here : {input_dir}")
    typer.echo(f"  Success source archive: {jobs_dir}/<job>/input")
    typer.echo(f"  Results are under     : {jobs_dir}/<job>/output")
    typer.echo(f"  Failed files go to    : {failed_dir}")
    typer.echo(f"  Pending image count   : {len(pending)}")
    if not pending:
        typer.echo(f"\n[WARN] No image files found. Add files to: {input_dir}")
    return input_dir, pending


def _quick_bg(config: dict) -> None:
    input_dir, pending = _show_task_help("bg", "BG / background removal")
    if not pending:
        return
    opts = config["bg"]
    batch_remove_cmd(input_dir, None, _workspace_root(), opts["recursive"], None, opts["backend"], opts.get("model"), opts["alpha_matting"], opts["retry_on_fail"], opts["continue_on_error"])


def _quick_sheet(config: dict) -> None:
    input_dir, pending = _show_task_help("sheet", "Sheets / character sheet splitting")
    if not pending:
        return
    opts = config["sheet"]
    batch_split_cmd(input_dir, None, _workspace_root(), opts["recursive"], None, opts["views"], opts["padding"], opts["min_area"], opts["merge_distance"], opts.get("normalize_size"), opts["threshold"], opts["debug"], opts["retry_on_fail"], opts["continue_on_error"])


def _quick_items(config: dict) -> None:
    input_dir, pending = _show_task_help("items", "Items / equipment-item extraction")
    if not pending:
        return
    opts = config["items"]
    batch_extract_cmd(input_dir, None, _workspace_root(), opts["recursive"], None, opts["padding"], opts["min_area"], opts["merge_distance"], opts["square_canvas"], opts.get("normalize_size"), opts["transparent_bg"], opts["threshold"], opts["debug"], opts["retry_on_fail"], opts["min_count"], opts["continue_on_error"])


def _quick_menu(config: dict) -> None:
    _show_config(config)
    bg_input, _, _ = _task_paths("bg")
    sheet_input, _, _ = _task_paths("sheet")
    items_input, _, _ = _task_paths("items")
    choice = _choose(
        "Quick run - choose task after placing files in the shown input folder",
        [
            ("bg", f"BG / background removal       -> put files in {bg_input}"),
            ("sheet", f"Sheets / character split     -> put files in {sheet_input}"),
            ("items", f"Items / equipment extraction -> put files in {items_input}"),
            ("back", "Back"),
        ],
    )
    if choice == "bg":
        _quick_bg(config)
    elif choice == "sheet":
        _quick_sheet(config)
    elif choice == "items":
        _quick_items(config)


@app.callback()
def menu_callback() -> None:
    _show_hint()
    config = _load_config()
    while True:
        choice = _choose("Main menu", [("quick", "Quick run"), ("config", "Config"), ("exit", "Exit")])
        if choice == "quick":
            _quick_menu(config)
        elif choice == "config":
            config = _config_menu(config)
        else:
            typer.echo("Bye.")
            return
