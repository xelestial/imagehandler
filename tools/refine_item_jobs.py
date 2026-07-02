from __future__ import annotations

import argparse
import json
from pathlib import Path

from imagehandler.item_refine import refine_item_outputs
from imagehandler.reports import BBox, OperationReport


def load_report(path: Path) -> OperationReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes = [BBox(*box) for box in data.get("boxes", [])]
    return OperationReport(
        ok=bool(data.get("ok", False)),
        operation=str(data.get("operation", "extract-items")),
        source=data.get("source"),
        backend=data.get("backend"),
        mode=data.get("mode"),
        warnings=list(data.get("warnings", [])),
        metrics=dict(data.get("metrics", {})),
        boxes=boxes,
        outputs=list(data.get("outputs", [])),
    )


def iter_manifests(root: Path) -> list[Path]:
    if root.is_file() and root.name == "manifest.json":
        return [root]
    return sorted(root.glob("**/manifest.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine ImageHandler item extraction job outputs.")
    parser.add_argument("path", type=Path, nargs="?", default=Path("workspace/items/jobs"))
    parser.add_argument("--min-area", type=int, default=120)
    args = parser.parse_args()

    manifests = iter_manifests(args.path)
    if not manifests:
        print(f"no manifests found: {args.path}")
        return 1

    changed = 0
    for manifest in manifests:
        report = load_report(manifest)
        before_items = len(report.outputs)
        before_dropped = int(report.metrics.get("postprocess_duplicate_partial_dropped", 0)) + int(report.metrics.get("postprocess_artifact_dropped", 0))
        report = refine_item_outputs(report, min_area=args.min_area)
        after_items = len(report.outputs)
        after_dropped = int(report.metrics.get("postprocess_duplicate_partial_dropped", 0)) + int(report.metrics.get("postprocess_artifact_dropped", 0))
        if after_items != before_items or after_dropped != before_dropped:
            changed += 1
        print(f"{manifest.parent}: {before_items} -> {after_items}")

    print(f"processed={len(manifests)} changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
