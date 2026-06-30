# BiSeNet face parsing ONNX workflow

ImageHandler uses a fixed local model path for BiSeNet face parsing:

```text
models/bisenet_face_parsing.onnx
```

The runtime and interactive menu do not read `IMAGEHANDLER_BISENET_ONNX` and do not require CLI model-path options. The menu only selects whether head refinement is `off`, `mediapipe`, or `bisenet`.

## Current state

This repository does not commit the ONNX model binary. The binary should be produced from a trusted BiSeNet face parsing checkpoint, verified with ONNX Runtime, then distributed as a release asset or copied to the fixed local path.

## Export from a local face-parsing.PyTorch checkout

Prepare a separate export environment where PyTorch is available. This avoids making PyTorch a runtime dependency of ImageHandler.

From inside the local face-parsing.PyTorch checkout, run:

```bash
python /path/to/imagehandler/tools/export_bisenet_from_face_parsing_repo.py \
  --checkpoint /path/to/79999_iter.pth \
  --output /path/to/imagehandler/models/bisenet_face_parsing.onnx \
  --meta /path/to/imagehandler/models/bisenet_face_parsing.onnx.json
```

The exporter expects `model.py` from face-parsing.PyTorch to be importable from the current working directory.

## Verify ONNX

After export:

```bash
cd /path/to/imagehandler
python tools/verify_bisenet_onnx.py \
  --model models/bisenet_face_parsing.onnx \
  --preview models/bisenet_face_parsing.preview.png \
  --json models/bisenet_face_parsing.verify.json
```

With a sample face image:

```bash
python tools/verify_bisenet_onnx.py \
  --model models/bisenet_face_parsing.onnx \
  --image /path/to/sample_face.png \
  --preview models/bisenet_face_parsing.preview.png \
  --json models/bisenet_face_parsing.verify.json
```

Check:

```text
input_shape is compatible, usually [1, 3, 512, 512]
output_shapes contains a 19-class logits tensor or label map
unique_labels includes plausible face parsing labels
preview image roughly separates face / hair / background
```

## Release asset workflow

Once the ONNX file is verified:

1. Upload `models/bisenet_face_parsing.onnx` as a project release asset.
2. Record its SHA256 from `models/bisenet_face_parsing.verify.json`.
3. Add the release URL and SHA256 to setup only after the asset is stable.
4. Setup can then download to `models/bisenet_face_parsing.onnx`, verify SHA256, and test ONNX Runtime loading.

Do not point setup to a PyTorch `.pth` file and rename it to `.onnx`. ONNX Runtime will not load that file.
