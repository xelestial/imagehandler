from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import shutil


@dataclass
class JobPaths:
    workspace_root: Path
    job_root: Path
    input_root: Path
    output_root: Path
    quality_root: Path
    logs_root: Path
    tmp_root: Path


def sanitize_job_name(name: str) -> str:
    text = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", name.strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "job"


def ensure_workspace_root(workspace_root: str | Path) -> Path:
    root = Path(workspace_root)
    for rel in ["inbox/bg", "inbox/sheets", "inbox/items", "jobs", "archive", "_reports"]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


def derive_job_name_from_input(input_path: str | Path) -> str:
    path = Path(input_path)
    return sanitize_job_name(path.stem)


def _unique_job_dir(jobs_root: Path, base_name: str) -> Path:
    candidate = jobs_root / sanitize_job_name(base_name)
    if not candidate.exists():
        return candidate
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return jobs_root / f"{sanitize_job_name(base_name)}_{suffix}"


def create_job_paths(workspace_root: str | Path, job_name: str | None = None, input_path: str | Path | None = None) -> JobPaths:
    root = ensure_workspace_root(workspace_root)
    jobs_root = root / "jobs"
    base_name = sanitize_job_name(job_name) if job_name else derive_job_name_from_input(input_path or "job")
    job_root = _unique_job_dir(jobs_root, base_name)
    input_root = job_root / "input"
    output_root = job_root / "output"
    quality_root = output_root / "quality"
    logs_root = output_root / "logs"
    tmp_root = job_root / "tmp"

    for rel in ["input/bg", "input/sheets", "input/items", "output/bg", "output/sheets", "output/items", "output/quality", "output/logs", "tmp"]:
        (job_root / rel).mkdir(parents=True, exist_ok=True)

    readme = job_root / "README.txt"
    if not readme.exists():
        readme.write_text(
            "\n".join(
                [
                    "ImageHandler job workspace",
                    "",
                    f"Job root: {job_root}",
                    "",
                    "Drop source files here:",
                    "  input/bg",
                    "  input/sheets",
                    "  input/items",
                    "",
                    "Find outputs here:",
                    "  output/bg",
                    "  output/sheets",
                    "  output/items",
                    "  output/quality",
                ]
            ),
            encoding="utf-8",
        )

    return JobPaths(
        workspace_root=root,
        job_root=job_root,
        input_root=input_root,
        output_root=output_root,
        quality_root=quality_root,
        logs_root=logs_root,
        tmp_root=tmp_root,
    )


def maybe_copy_input_to_job(job_paths: JobPaths, task_group: str, source_path: str | Path) -> Path:
    src = Path(source_path)
    dst_dir = job_paths.input_root / task_group
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    try:
        if src.resolve() != dst.resolve() and not dst.exists():
            shutil.copy2(src, dst)
    except Exception:
        # Non-fatal archival copy.
        pass
    return dst


def resolve_output_for_task(
    task_group: str,
    input_path: str | Path,
    output: str | Path | None,
    workspace: str | Path | None,
    job_name: str | None = None,
) -> tuple[Path, JobPaths | None]:
    if output is not None:
        return Path(output), None
    workspace_root = Path(workspace or "workspace")
    job_paths = create_job_paths(workspace_root, job_name=job_name, input_path=input_path)
    maybe_copy_input_to_job(job_paths, task_group, input_path)
    stem = Path(input_path).stem
    if task_group == "bg":
        return job_paths.output_root / "bg" / f"{stem}.png", job_paths
    if task_group == "sheets":
        return job_paths.output_root / "sheets" / stem, job_paths
    if task_group == "items":
        return job_paths.output_root / "items" / stem, job_paths
    if task_group == "quality":
        return job_paths.output_root / "quality" / f"{stem}.judge.json", job_paths
    return job_paths.output_root / stem, job_paths
