# Source map

This repository extracts the detection + SAM3 mask path from HaWoR commit
`579253db1787b306161e81444170178cd6628e37`.

| Standalone file | Production source |
|---|---|
| `detection.py` | `pipeline.py::save_yolo_detector_area_max_bboxes` |
| `detection_payload.py` | `detection_payload.py` |
| `sam3_point.py` | `sam3_point.py` |
| `sam3_helpers.py` | the helper surface used from `sam_anchor.py` and `sam_repropagate.py` |
| `ego_candidate_score.py` | `ego_candidate_score.py` |

The default config matches the current gated Egocentric-10K T production route:
batched YOLO `predict` with batch `16`, confidence `0.2`, per-frame/per-side
largest-area selection, SAM3 seed threshold `0.75`, 5-second windows, `joint`
tracking, `point` init, `box_below` prompt geometry, `single_anchor`
propagation, NPZ mask archives, and the production bbox-only ego gate. The
live script fingerprint and complete parameter mapping are recorded in
`PRODUCTION_DEFAULTS.md`.

Standalone-only glue does three things that the larger HaWoR pipeline used to
do around those functions:

1. decode/normalize input frames;
2. resize frames and boxes together for the SAM3 1008-pixel input boundary;
3. restore binary masks to original decoded-frame coordinates before writing
   the handoff masks and QA overlay.

The CUDA-invalid automatic retry from the production worker is intentionally
not carried into the handoff CLI. A runtime failure stops at the failing call.
