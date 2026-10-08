# InfiniHand inference

Inference-only extraction: **SAM3 + tracking (default) -> InfiniHand MANO and camera prediction -> sparse DROID BA at stride 20 -> world-space hands**.

An optional built-in mask head uses `--mask-source head --mask-threshold 0.9`. Both paths share the same `infinihand_v1.pt`. No training loop, optimizer or dataset evaluation pipeline is provided.

## Weights

`weights/infinihand_v1.pt` is a **plain PyTorch state_dict**, merging stage1_v6 geometry/mask/backbone with the 190k camera head. It contains 1,912 tensors: 69 camera tensors and 1,843 unchanged stage1_v6 tensors. It contains no optimizer, step, scheduler or training config. Architecture settings live separately in `configs/infinihand_v1.json`.

- Size: 5,588,232,558 bytes.
- SHA256: `1ce8c8846537e984e3be1faa681347079a2d3867abf604bb4c6cd80527c7e2a4`.
- Weights are **not committed to Git**. Copy the separately exported `infinihand_v1.pt` into `weights/`.
- Place licensed MANO assets and `mano_mean_params.npz` in `weights/mano/`, following WiLoR's `mano_data` layout. MANO terms apply separately.
- Default tracking additionally needs a detector checkpoint, the gated SAM3 checkpoint and installed SAM3 source. See `third_party/ego_step2/README.md`.
- Sparse BA needs `weights/droid.pth` from HaWoR/DROID-SLAM.

## Environment

Linux, CUDA and a compatible PyTorch installation are required. Install PyTorch first for your GPU, then:

```bash
pip install -r requirements.txt
# Builds the vendored DROID and LieTorch CUDA extensions:
(cd third_party/droid && pip install --no-build-isolation .)
```

SAM3 may use its **own Python environment**; install SAM3 and the requirements from `third_party/ego_step2/requirements.txt` there. Pass `--sam3-python` and `--sam3-source` to select it. BA can likewise use a compatible CUDA-extension environment through `--ba-python`.

## Run

Provide original-image intrinsics, e.g. `camera.json`:

```json
{"fx": 700.0, "fy": 700.0, "cx": 640.0, "cy": 360.0, "image_width": 1280, "image_height": 720}
```

```bash
python -m infinihand.cli \
  --video input.mp4 --intrinsics camera.json --output outputs/example \
  --checkpoint weights/infinihand_v1.pt \
  --sam3-source /path/to/sam3 --sam3-checkpoint weights/sam3.pt \
  --detector weights/detector.pt --droid-weights weights/droid.pth
```

Optional internal mask head:

```bash
python -m infinihand.cli --video input.mp4 --intrinsics camera.json \
  --output outputs/head --mask-source head --mask-threshold 0.9
```

`--no-ba` exports the camera-head trajectory without refinement. `--ba-stride` defaults to **20**; the final frame is also retained. Use a fresh output directory on each run. Intrinsics must match the decoded image dimensions; the code does not silently guess focal length or correct mismatched calibration.

## Outputs and conventions

- `camera_space.npz`: joints, vertices, global orientation, MANO pose/shape, translation, validity and camera encoding.
- `trajectory.npz`: BA-refined camera-to-first-camera trajectory, initial trajectory and selected keyframes (when BA enabled).
- `world_space.npz`: transformed geometry/MANO global orientation and translation, plus unchanged local pose/shape.
- `SUCCESS.json`: completed-frame count, selected mask path, timing and memory-contract receipt.
- SAM3 outputs preserve the original per-side masks and metadata.

Hand slots are **right, left**. Pose/orientation matrices use the saved model's conventions. Camera-to-world transforms map column vectors into the first predicted camera frame. World outputs retain the camera head's learned translation scale; this wrapper performs no GT alignment or independent metric-scale recovery. Monocular scale accuracy is not guaranteed.

The streaming backbone retains four anchor frames and sixteen recent dense frames. This wrapper stages decoded frames in memory and currently supports at most 8,192 frames per invocation; it is not an unbounded video loader. BA operates after prediction, using the existing fixed-stride backend with 7+12 iterations, hand-box exclusion, gauge normalization tracking and interpolation of pose corrections. It is **not** the asynchronous binary keyframe-pool scheduler. No throughput claim is made for this wrapper.

## Provenance

- EgoPreprocess step2: `cskrren/EgoPreprocess`, commit `2e33f7647d3f32248c207e601b1b6adb534684ae`; point seeds + joint propagation, not independent per-frame segmentation.
- InfiniHand: exported runtime from the validated stage1_v6 dual-stream inference implementation.
- Sparse BA: the existing camera190k + BA20 inference backend; training/evaluation orchestration removed.

Third-party code retains its upstream licenses/notices. See `NOTICE.md`. Checkpoints and MANO assets have their own terms.

## Validation

Both routes were executed end to end on an 81-frame HOI4D clip: SAM3 video tracking and internal mask head (threshold 0.9). Each used the merged weight file, camera head, five BA keyframes `[0, 20, 40, 60, 80]`, and exported finite `[81, 2, 778, 3]` world vertices. The first-camera transform was checked against identity. Geometry unit tests cover pose inversion and recovery of a known translation with fixed-depth LSP. See `configs/smoke_validation.json`; this is a functional smoke test, not benchmark accuracy validation.
