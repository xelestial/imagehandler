from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import shutil


TASK_DIRS = {
    "bg": "bg",
    "sheets": "sheets",
    "items": "items",
    "quality": "quality",
}


@dataclass
class JobPaths:
    workspace_root: Path
    task_root: Path
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


def task_dir_name(task_group: str) -> str:
    return TASK_DIRS.get(task_group, sanitize_job_name(task_group))


def ensure_workspace_root(workspace_root: str | Path) -> Path:
    root = Path(workspace_root)
    for task in ["bg", "sheets", "items"]:
        for rel in ["input", "complete", "jobs"]:
            (root / task / rel).mkdir(parents=True, exist_ok=True)
    for rel in ["quality/input", "quality/complete", "quality/jobs", "archive", "_reports"]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


def derive_job_name_from_input(input_path: str | Path) -> str:
    return sanitize_job_name(Path(input_path).stem)


def _unique_job_dir(jobs_root: Path, base_name: str) -> Path:
    base = sanitize_job_name(base_name)
    candidate = jobs_root / base
    if not candidate.exists():
        return candidate
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return jobs_root / f"{base}_{suffix}"


def create_job_paths(
    workspace_root: str | Path,
    task_group: str,
    job_name: str | None = None,
    input_path: str | Path | None = None,
) -> JobPaths:
    root = ensure_workspace_root(workspace_root)
    task_root = root / task_dir_name(task_group)
    jobs_root = task_root / "jobs"
    base_name = sanitize_job_name(job_name) if job_name else derive_job_name_from_input(input_path or "job")
    job_root = _unique_job_dir(jobs_root, base_name)

    input_root = job_root / "input"
    output_root = job_root / "output"
    quality_root = job_root / "quality"
    logs_root = job_root / "logs"
    tmp_root = job_root / "tmp"

    for folder in [input_root, output_root, quality_root, logs_root, tmp_root]:
        folder.mkdir(parents=True, exist_ok=True)

    readme = job_root / "README.txt"
    if not readme.exists():
        readme.write_text(
            "\n".join(
                [
                    "ImageHandler task-first job workspace",
                    "",
                    f"Task root: {task_root}",
                    f"Job root: {job_root}",
                    "",
                    "Input copy:",
                    "  input/",
                    "",
                    "Outputs:",
                    "  output/",
                    "",
                    "Quality reports:",
                    "  quality/",
                    "",
                    "Logs:",
                    "  logs/",
                ]
            ),
            encoding="utf-8",
        )

    return JobPaths(
        workspace_root=root,
        task_root=task_root,
        job_root=job_root,
        input_root=input_root,
        output_root=output_root,
        quality_root=quality_root,
        logs_root=logs_root,
        tmp_root=tmp_root,
    )


def maybe_copy_input_to_job(job_paths: JobPaths, source_path: str | Path) -> Path:
    src = Path(source_path)
    dst = job_paths.input_root / src.name
    try:
        if src.is_file() and src.resolve() != dst.resolve() and not dst.exists():
            shutil.copy2(src, dst)
    except Exception:
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
    job_paths = create_job_paths(
        workspace_root=workspace_root,
        task_group=task_group,
        job_name=job_name,
        input_path=input_path,
    )
    maybe_copy_input_to_job(job_paths, input_path)

    stem = Path(input_path).stem
    if task_group == "bg":
        return job_paths.output_root / f"{stem}.png", job_paths
    if task_group in {"sheets", "items"}:
        return job_paths.output_root, job_paths
    if task_group == "quality":
        return job_paths.quality_root / f"{stem}.judge.json", job_paths
    return job_paths.output_root / stem, job_paths
