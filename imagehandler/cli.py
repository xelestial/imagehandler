from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .judge import judge_path as judge_path_impl

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("judge")
def judge_cmd(
    target: Path = typer.Argument(..., exists=True, readable=True, help="Image file or output directory to judge."),
    task: str = typer.Option("auto", help="auto, generic, remove-bg, split-sheet, extract-items"),
    expected_count: Optional[int] = typer.Option(None, help="Expected number of outputs, e.g. 4 for a character sheet."),
    min_count: int = typer.Option(1, help="Minimum acceptable number of images when judging a directory."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Judge JSON path."),
    debug: bool = typer.Option(False, help="Reserved for debug outputs."),
    alpha_required: Optional[bool] = typer.Option(None, help="Force alpha/transparency requirement on or off."),
):
    """Judge processed image quality and write PASS/WARN/FAIL report."""
    report = judge_path_impl(
        target=target,
        task=task,
        expected_count=expected_count,
        min_count=min_count,
        output=output,
        debug=debug,
        alpha_required=alpha_required,
    )
    _print_quality_report(report)


def _print_quality_report(report) -> None:
    verdict = report["verdict"] if isinstance(report, dict) else report.verdict
    score = report["score"] if isinstance(report, dict) else report.score
    task = report["task"] if isinstance(report, dict) else report.task
    target = report["target"] if isinstance(report, dict) else report.target
    failures = report.get("failures", []) if isinstance(report, dict) else report.failures
    warnings = report.get("warnings", []) if isinstance(report, dict) else report.warnings
    color = "green" if verdict == "PASS" else "yellow" if verdict == "WARN" else "red"
    console.print(f"[{color}][bold]{verdict}[/bold][/{color}] judge {task} score={score}")
    if failures:
        console.print("[red]failures:[/red]")
        for item in failures:
            console.print(f"  - {item}")
    if warnings:
        console.print("[yellow]warnings:[/yellow]")
        for item in warnings:
            console.print(f"  - {item}")
    console.print(f"target: {target}")


if __name__ == "__main__":
    app()
