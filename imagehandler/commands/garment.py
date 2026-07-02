from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from imagehandler.person_clothing import separate_person_clothing
from imagehandler.workspace import resolve_output_for_task
from .common import print_job_paths, print_operation_report

app = typer.Typer(no_args_is_help=True, help="Garment parsing commands.")


@app.command("extract")
def extract_cmd(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o"),
    workspace: Optional[Path] = typer.Option(None),
    job: Optional[str] = typer.Option(None),
    debug: bool = typer.Option(True),
):
    out_dir, job_paths = resolve_output_for_task("clothing", input_path, output_dir, workspace, job)
    report = separate_person_clothing(input_path=input_path, output_dir=out_dir, debug=debug)
    print_operation_report(report)
    print_job_paths(job_paths)
