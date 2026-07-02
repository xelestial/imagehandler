from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


TASK_DIRS = {
    "bg": "bg",
    "sheets": "sheets",
    "items": "items",
    "clothing": "clothing",
    "garment": "clothing",
}


@dataclass
class JobPaths:
    workspace_root: Path
    task_root: Path
    job_root: Path
    input_root: Path
    output_root: Path
    failed_root: Path
    reports_root: Path

    @property
    def quality_root(self) -> Path:
        return self.reports_root

    @property
    def logs_root(self) -> Path:
        return self.reports_root

    @property
    def tmp_root(self) -> Path:
        return self.job_root / "tmp"


def sanitize_job_name(name: str) -> str:
    text = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", name.strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "job"


def task_dir_name(task_group: str) -> str:
    return TASK_DIRS.get(task_group, sanitize_job_name(task_group))


def ensure_workspace_root(workspace_root: str | Path) -> Path:
    root = Path(workspace_root)
    for task in ["bg", "sheets", "items", "clothing"]:
        for rel in ["input", "jobs", "failed"]:
            (root / task / rel).mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
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
    task = task_dir_name(task_group)
    task_root = root / task
    jobs_root = task_root / "jobs"
    failed_root = task_root / "failed"

    base_name = sanitize_job_name(job_name) if job_name else derive_job_name_from_input(input_path or "job")
    job_root = _unique_job_dir(jobs_root, base_name)
    input_root = job_root / "input"
    output_root = job_root
    reports_root = job_root

    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    return JobPaths(
        workspace_root=root,
        task_root=task_root,
        job_root=job_root,
        input_root=input_root,
        output_root=output_root,
        failed_root=failed_root,
        reports_root=reports_root,
    )


def resolve_output_for_task(
    task_group: str,
    input_path: str | Path,
    output: str | Path | None,
    workspace: str | Path | None,
    job_name: str | None = None,
) -> tuple[Path, JobPaths | None]:
    if output is not None:
        return Path(output), None

    workspace_root = ensure_workspace_root(workspace or "workspace")
    stem = Path(input_path).stem

    if task_group == "quality":
        reports_root = workspace_root / "reports"
        reports_root.mkdir(parents=True, exist_ok=True)
        return reports_root / f"{stem}.judge.json", None

    job_paths = create_job_paths(
        workspace_root=workspace_root,
        task_group=task_group,
        job_name=job_name,
        input_path=input_path,
    )

    if task_group == "bg":
        return job_paths.output_root / f"{stem}.png", job_paths
    if task_group in {"sheets", "items", "clothing", "garment"}:
        return job_paths.output_root, job_paths
    return job_paths.output_root / stem, job_paths
