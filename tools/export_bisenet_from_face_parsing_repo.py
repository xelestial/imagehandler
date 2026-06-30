#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_state_dict(raw: dict[str, Any]) -> dict[str, Any]:
    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    elif "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key)
        for prefix in ("module.", "model.", "net."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export BiSeNet face-parsing checkpoint to ONNX. "
            "Run this script from inside a local face-parsing.PyTorch checkout, "
            "or pass --source-repo to the checkout root."
        )
    )
    parser.add_argument("--source-repo", type=Path, default=Path.cwd(), help="face-parsing.PyTorch checkout root containing model.py. Default: current directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/bisenet_face_parsing.onnx"))
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--meta", type=Path, default=Path("models/bisenet_face_parsing.onnx.json"))
    args = parser.parse_args()

    source_repo = args.source_repo.expanduser().resolve()
    model_py = source_repo / "model.py"
    if not model_py.is_file():
        raise FileNotFoundError(f"model.py not found in source repo: {source_repo}")
    sys.path.insert(0, str(source_repo))

    import torch
    from model import BiSeNet  # type: ignore

    net = BiSeNet(n_classes=args.num_classes)
    raw = torch.load(str(args.checkpoint), map_location="cpu")
    state = clean_state_dict(raw)
    missing, unexpected = net.load_state_dict(state, strict=False)
    net.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, x):  # noqa: ANN001
            out = self.inner(x)
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

    wrapper = Wrapper(net).eval()
    dummy = torch.randn(1, 3, args.input_size, args.input_size, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(args.output),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    report = {
        "output": str(args.output),
        "sha256": sha256_file(args.output),
        "size_bytes": args.output.stat().st_size,
        "source_repo": str(source_repo),
        "checkpoint": str(args.checkpoint),
        "input_shape": [1, 3, args.input_size, args.input_size],
        "output_name": "logits",
        "num_classes": args.num_classes,
        "opset": args.opset,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
