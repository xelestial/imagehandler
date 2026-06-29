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
    "bg": {"single": False, "recursive": True, "backend": "auto", "model": None, "alpha_matting": False, "retry_on_fail": True, "continue_on_error": True},
    "sheet": {"single": False, "recursive": True, "views": 4, "padding": 24, "min_area": 1000, "merge_distance": 24, "normalize_size": None, "threshold": 28.0, "debug": True, "retry_on_fail": True, "continue_on_error": True, "accept_verdict": "PASS", "min_score": 85.0},
    "items": {"single": False, "recursive": True, "padding": 16, "min_area": 120, "merge_distance": 12, "square_canvas": False, "normalize_size": None, "transparent_bg": False, "threshold": 28.0, "debug": True, "retry_on_fail": True, "min_count": 1, "continue_on_error": True, "accept_verdict": "PASS", "min_score": 85.0},
    "quality": {"single": False, "recursive": True, "task": "auto", "alpha_required": None, "continue_on_error": True, "expected_count": None, "min_count": 1, "debug": False},
}

PROFILES = {
    "fast": {"bg": {"retry_on_fail": False}, "sheet": {"debug": False, "retry_on_fail": False}, "items": {"debug": False, "retry_on_fail": False}},
    "balanced": {"bg": {"retry_on_fail": True}, "sheet": {"debug": True, "retry_on_fail": True}, "items": {"debug": True, "retry_on_fail": True}},
    "high_quality": {"bg": {"retry_on_fail": True, "alpha_matting": True}, "sheet": {"debug": True, "retry_on_fail": True, "min_score": 90.0}, "items": {"debug": True, "retry_on_fail": True, "min_score": 90.0}},
}


def _workspace_root() -> Path:
    return Path(os.environ.get("IMAGEHANDLER_WORKSPACE", "workspace"))


def _ensure_workspace() -> Path:
    root = _workspace_root()
    for task in ["bg", "sheets", "items", "quality"]:
        for rel in ["input", "complete", "jobs"]:
            (root / task / rel).mkdir(parents=True, exist_ok=True)
    (root / "archive").mkdir(parents=True, exist_ok=True)
    (root / "_reports").mkdir(parents=True, exist_ok=True)
    return root


def _deep_update(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
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


def _task_input(task: str) -> Path:
    root = _ensure_workspace()
    if task == "bg":
        return root / "bg" / "input"
    if task == "sheet":
        return root / "sheets" / "input"
    if task == "items":
        return root / "items" / "input"
    return root / "quality" / "input"


def _list_images(path: Path, recursive: bool = True) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    pattern = "**/*" if recursive else "*"
    return sorted([p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "complete" not in p.parts])


def _one_image_or_warn(folder: Path) -> Path | None:
    files = _list_images(folder, recursive=True)
    if not files:
        typer.echo(f"\n[WARN] No image files found in: {folder}")
        return None
    if len(files) > 1:
        typer.echo(f"Using first image: {files[0]} ({len(files)} found)")
    return files[0]


def _show_hint() -> None:
    root = _ensure_workspace()
    typer.echo("\nTask-first workspace:")
    typer.echo(f"  BG input      : {root / 'bg' / 'input'}")
    typer.echo(f"  Sheets input  : {root / 'sheets' / 'input'}")
    typer.echo(f"  Items input   : {root / 'items' / 'input'}")
    typer.echo(f"  Results       : {root}/<task>/jobs/<job>/output")
    typer.echo(f"  Complete      : {root}/<task>/complete")


def _show_config(config: dict) -> None:
    typer.echo("\nCurrent config")
    typer.echo(f"  Profile: {config.get('profile')}")
    typer.echo(f"  BG    : {'single' if config['bg']['single'] else 'batch'}, retry={config['bg']['retry_on_fail']}")
    typer.echo(f"  Sheet : {'single' if config['sheet']['single'] else 'batch'}, views={config['sheet']['views']}, retry={config['sheet']['retry_on_fail']}")
    typer.echo(f"  Items : {'single' if config['items']['single'] else 'batch'}, retry={config['items']['retry_on_fail']}")


def _config_menu(config: dict) -> dict:
    while True:
        _show_config(config)
        choice = _choose("Config", [("profile", "Profile"), ("mode", "Single/batch mode"), ("reset", "Reset defaults"), ("back", "Back")])
        if choice == "profile":
            profile = _choose("Profile", [("fast", "Fast"), ("balanced", "Balanced"), ("high_quality", "High quality")])
            _apply_profile(config, profile)
            _save_config(config)
        elif choice == "mode":
            task = _choose("Task", [("bg", "BG"), ("sheet", "Sheets"), ("items", "Items"), ("quality", "Quality")])
            mode = _choose("Mode", [("batch", "Batch folder"), ("single", "Single file")])
            config[task]["single"] = mode == "single"
            _save_config(config)
        elif choice == "reset":
            config = json.loads(json.dumps(DEFAULT_CONFIG))
            _save_config(config)
        else:
            return config


def _quick_bg(config: dict) -> None:
    opts = config["bg"]
    folder = _task_input("bg")
    if opts["single"]:
        path = _one_image_or_warn(folder)
        if path is None:
            return
        remove_cmd(path, None, _workspace_root(), None, opts["backend"], opts.get("model"), opts["alpha_matting"], False, False, 0.0, opts["retry_on_fail"], "PASS", 85.0)
    else:
        batch_remove_cmd(folder, None, _workspace_root(), opts["recursive"], None, opts["backend"], opts.get("model"), opts["alpha_matting"], opts["retry_on_fail"], opts["continue_on_error"])


def _quick_sheet(config: dict) -> None:
    opts = config["sheet"]
    folder = _task_input("sheet")
    if opts["single"]:
        path = _one_image_or_warn(folder)
        if path is None:
            return
        split_cmd(path, None, _workspace_root(), None, opts["views"], opts["padding"], opts["min_area"], opts["merge_distance"], opts.get("normalize_size"), opts["threshold"], opts["debug"], opts["retry_on_fail"], opts["accept_verdict"], opts["min_score"])
    else:
        batch_split_cmd(folder, None, _workspace_root(), opts["recursive"], None, opts["views"], opts["padding"], opts["min_area"], opts["merge_distance"], opts.get("normalize_size"), opts["threshold"], opts["debug"], opts["retry_on_fail"], opts["continue_on_error"])


def _quick_items(config: dict) -> None:
    opts = config["items"]
    folder = _task_input("items")
    if opts["single"]:
        path = _one_image_or_warn(folder)
        if path is None:
            return
        extract_cmd(path, None, _workspace_root(), None, opts["padding"], opts["min_area"], opts["merge_distance"], opts["square_canvas"], opts.get("normalize_size"), opts["transparent_bg"], opts["threshold"], opts["debug"], opts["retry_on_fail"], opts["accept_verdict"], opts["min_score"], opts["min_count"])
    else:
        batch_extract_cmd(folder, None, _workspace_root(), opts["recursive"], None, opts["padding"], opts["min_area"], opts["merge_distance"], opts["square_canvas"], opts.get("normalize_size"), opts["transparent_bg"], opts["threshold"], opts["debug"], opts["retry_on_fail"], opts["min_count"], opts["continue_on_error"])


def _quick_quality(config: dict) -> None:
    opts = config["quality"]
    target = _workspace_root() / "quality" / "input"
    if opts["single"]:
        path = _one_image_or_warn(target)
        if path is None:
            return
        judge_cmd(path, opts["task"], opts.get("expected_count"), opts["min_count"], None, _workspace_root(), None, opts["debug"], opts.get("alpha_required"))
    else:
        batch_judge_cmd(target, None, _workspace_root(), opts["recursive"], None, opts["task"], opts.get("alpha_required"), opts["continue_on_error"])


def _quick_menu(config: dict) -> None:
    _show_config(config)
    choice = _choose("Quick run", [("bg", "BG"), ("sheet", "Sheets"), ("items", "Items"), ("quality", "Quality"), ("back", "Back")])
    if choice == "bg":
        _quick_bg(config)
    elif choice == "sheet":
        _quick_sheet(config)
    elif choice == "items":
        _quick_items(config)
    elif choice == "quality":
        _quick_quality(config)


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
