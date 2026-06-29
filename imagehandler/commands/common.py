from __future__ import annotations

from rich.console import Console

console = Console()


def print_operation_report(report) -> None:
    status = "OK" if getattr(report, "ok", False) else "CHECK"
    console.print(f"[bold]{status}[/bold] {report.operation}")
    if getattr(report, "backend", None):
        console.print(f"backend: {report.backend}")
    if getattr(report, "mode", None):
        console.print(f"mode: {report.mode}")
    if getattr(report, "outputs", None):
        console.print("outputs:")
        for path in report.outputs:
            console.print(f"  - {path}")
    if getattr(report, "warnings", None):
        console.print("[yellow]warnings:[/yellow]")
        for item in report.warnings:
            console.print(f"  - {item}")


def print_quality_report(report) -> None:
    color = "green" if report.verdict == "PASS" else "yellow" if report.verdict == "WARN" else "red"
    console.print(f"[{color}][bold]{report.verdict}[/bold][/{color}] judge {report.task} score={report.score}")
    if report.failures:
        console.print("[red]failures:[/red]")
        for item in report.failures:
            console.print(f"  - {item}")
    if report.warnings:
        console.print("[yellow]warnings:[/yellow]")
        for item in report.warnings:
            console.print(f"  - {item}")
    console.print(f"target: {report.target}")


def print_fallback_summary(summary) -> None:
    color = "green" if summary.accepted else "yellow"
    console.print(f"[{color}]fallback selected[/{color}]: {summary.selected_attempt} score={summary.selected_score} verdict={summary.selected_verdict}")
    console.print("attempts:")
    for attempt in summary.attempts:
        state = "ACCEPTED" if attempt.accepted else "tried"
        console.print(f"  - {attempt.name} -> {attempt.verdict} {attempt.score} ({state})")


def _print_moved(title: str, moved: list[str], color: str) -> None:
    if not moved:
        return
    console.print(f"[{color}]{title}:[/{color}] {len(moved)} file(s)")
    preview = moved[:5]
    for item in preview:
        console.print(f"  - {item}")
    if len(moved) > len(preview):
        console.print(f"  ... {len(moved) - len(preview)} more")


def print_batch_result(result) -> None:
    color = "green" if result.ok else "red"
    console.print(f"[{color}][bold]{result.operation}[/bold][/{color}] total={result.total} ok={result.succeeded} failed={result.failed}")
    _print_moved("moved to complete", getattr(result, "moved_to_complete", []) or [], "cyan")
    _print_moved("moved to failed", getattr(result, "moved_to_failed", []) or [], "yellow")
    if result.errors:
        console.print("[red]errors:[/red]")
        for item in result.errors:
            console.print(f"  - {item}")


def print_job_paths(job_paths) -> None:
    if job_paths is None:
        return
    console.print("[cyan]output job folder:[/cyan] " + str(job_paths.output_root))
