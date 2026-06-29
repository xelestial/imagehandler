from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from imagehandler.batch import (
    BatchResult,
    iter_image_files,
    move_input_to_failed,
    move_input_to_job_input,
    relative_output_file,
)
from imagehandler.bg_remove import remove_background
from imagehandler.fallback import remove_background_with_fallback
from imagehandler.workspace import resolve_output_for_task

from .common import print_batch_result, print_fallback_summary, print_job_paths, print_operation_report

app = typer.Typer(no_args_is_help=True, help="Background removal menu.")


@app.command("remove")
def remove_cmd(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output PNG path. If omitted, use workspace/bg/jobs/<job>/output/."),
    workspace: Optional[Path] = typer.Option(None, help="Workspace root used when --output is omitted. Default: ./workspace"),
    job: Optional[str] = typer.Option(None, help="Optional job folder name. If omitted, use the input filename."),
    backend: str = typer.Option("auto", help="auto, rembg, transparent, classical"),
    model: Optional[str] = typer.Option(None, help="rembg model name."),
    alpha_matting: bool = typer.Option(False, help="Enable rembg alpha matting."),
    mask_only: bool = typer.Option(False, help="Write mask only."),
    no_postprocess: bool = typer.Option(False, help="Disable mask cleanup."),
    feather: float = typer.Option(0.0, help="Gaussian blur radius for alpha feathering."),
    retry_on_fail: bool = typer.Option(False, help="Try fallback backends if quality check fails."),
    accept_verdict: str = typer.Option("PASS", help="PASS or WARN."),
    min_score: float = typer.Option(85.0, help="Minimum accepted quality score."),
):
    output_path, job_paths = resolve_output_for_task("bg", input_path, output, workspace, job)
    if retry_on_fail:
        report, summary = remove_background_with_fallback(
            input_path=input_path,
            output_path=output_path,
            backend=backend,
            model=model,
            alpha_matting=alpha_matting,
            mask_only=mask_only,
            postprocess=not no_postprocess,
            feather=feather,
            accept_verdict=accept_verdict,
            min_score=min_score,
        )
        print_operation_report(report)
        print_fallback_summary(summary)
        print_job_paths(job_paths)
        return
    report = remove_background(
        input_path=input_path,
        output_path=output_path,
        backend=backend,
        model=model,
        alpha_matting=alpha_matting,
        mask_only=mask_only,
        postprocess=not no_postprocess,
        feather=feather,
    )
    print_operation_report(report)
    print_job_paths(job_paths)


@app.command("batch-remove")
def batch_remove_cmd(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Image file or input directory."),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory. If omitted, use workspace/bg/jobs/<job>/output/ per file."),
    workspace: Optional[Path] = typer.Option(None, help="Workspace root used when --output is omitted. Default: ./workspace"),
    recursive: bool = typer.Option(False, help="Search recursively."),
    pattern: Optional[str] = typer.Option(None, help="Glob pattern. Example: '**/*.png'"),
    backend: str = typer.Option("auto"),
    model: Optional[str] = typer.Option(None),
    alpha_matting: bool = typer.Option(False),
    retry_on_fail: bool = typer.Option(False),
    continue_on_error: bool = typer.Option(True, help="Keep processing after a failed file."),
):
    files = iter_image_files(input_path, recursive=recursive, pattern=pattern)
    result = BatchResult(operation="bg batch-remove", total=len(files))
    root = input_path if input_path.is_dir() else input_path.parent
    output_jobs = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        job_paths = None
        if output_dir is None:
            dst, job_paths = resolve_output_for_task("bg", src, None, workspace, None)
            output_jobs.append(str(job_paths.output_root) if job_paths else str(dst.parent))
        else:
            dst = relative_output_file(src, root, output_dir, suffix=".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if retry_on_fail:
                remove_background_with_fallback(src, dst, backend=backend, model=model, alpha_matting=alpha_matting)
            else:
                remove_background(src, dst, backend=backend, model=model, alpha_matting=alpha_matting)
            result.succeeded += 1
            result.outputs.append(str(dst))
            if job_paths is not None:
                moved = move_input_to_job_input(src, root, job_paths.input_root)
                if moved is not None:
                    result.moved_to_job_input.append(str(moved))
        except Exception as exc:
            result.failed += 1
            result.errors.append(f"{src}: {exc}")
            moved = move_input_to_failed(src, root)
            if moved is not None:
                result.moved_to_failed.append(str(moved))
            if not continue_on_error:
                raise
    print_batch_result(result)
    for item in output_jobs:
        typer.echo(f"output job folder: {item}")
