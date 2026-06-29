from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from imagehandler.batch import BatchResult, iter_image_files, relative_output_file
from imagehandler.judge import judge_path
from imagehandler.workspace import resolve_output_for_task

from .common import print_batch_result, print_job_paths, print_quality_report

app = typer.Typer(no_args_is_help=True, help="Quality judging menu.")


@app.command("judge")
def judge_cmd(
    target: Path = typer.Argument(..., exists=True, readable=True),
    task: str = typer.Option("auto", help="auto, generic, remove-bg, split-sheet, extract-items"),
    expected_count: Optional[int] = typer.Option(None),
    min_count: int = typer.Option(1),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Judge JSON path. If omitted, auto-create a job folder under workspace/jobs/."),
    workspace: Optional[Path] = typer.Option(None, help="Workspace root used when --output is omitted. Default: ./workspace"),
    job: Optional[str] = typer.Option(None, help="Optional job folder name. If omitted, use the input filename. If the name already exists, a date suffix is appended."),
    debug: bool = typer.Option(False),
    alpha_required: Optional[bool] = typer.Option(None),
):
    out_path, job_paths = resolve_output_for_task("quality", target, output, workspace, job)
    report = judge_path(
        target=target,
        task=task,
        expected_count=expected_count,
        min_count=min_count,
        output=out_path,
        debug=debug,
        alpha_required=alpha_required,
    )
    print_quality_report(report)
    print_job_paths(job_paths)


@app.command("batch-judge")
def batch_judge_cmd(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Directory for judge JSON files. If omitted, auto-create one job folder per file under workspace/jobs/."),
    workspace: Optional[Path] = typer.Option(None, help="Workspace root used when --output is omitted. Default: ./workspace"),
    recursive: bool = typer.Option(False),
    pattern: Optional[str] = typer.Option(None),
    task: str = typer.Option("auto"),
    alpha_required: Optional[bool] = typer.Option(None),
    continue_on_error: bool = typer.Option(True),
):
    files = iter_image_files(input_path, recursive=recursive, pattern=pattern)
    result = BatchResult(operation="quality batch-judge", total=len(files))
    root = input_path if input_path.is_dir() else input_path.parent
    auto_job_outputs = []
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        out = None
        if output_dir:
            out = relative_output_file(src, root, output_dir, suffix=".judge.json")
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out, job_paths = resolve_output_for_task("quality", src, None, workspace, None)
            auto_job_outputs.append(str(job_paths.job_root) if job_paths else str(out.parent))
        try:
            judge_path(src, task=task, output=out, alpha_required=alpha_required)
            result.succeeded += 1
            result.outputs.append(str(out or src.with_name(f"{src.stem}.judge.json")))
        except Exception as exc:
            result.failed += 1
            result.errors.append(f"{src}: {exc}")
            if not continue_on_error:
                raise
    print_batch_result(result)
    if auto_job_outputs:
        for item in auto_job_outputs:
            typer.echo(f"job folder: {item}")
