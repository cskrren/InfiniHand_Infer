# Production default mapping

Live-checked on 2026-07-16 against:

```text
/mnt/petrelfs/songkaiwen/projects/HaWoR/scripts/ego4d_sam3_hawor/unified_env/egocentric10k_t_8gpu_gated_production.sbatch
```

Production script SHA-256:

```text
52ffddbc8a01bf411e63f0eddad0e84657f2f96bcf212bbeb47056708ecfd476
```

The bundled SAM3 source is the exact production worktree:

```text
commit: 8e451d5eb43c817b64ae7577fb7b9ae223db88a9
tracked diff SHA-256: 22c88d8ffdaceff94a0030252a71a5ce70ab15162dd9c64c1cb7a791c7a41218
```

The tracked diff adds runtime signature filtering in
`sam3/model/sam3_base_predictor.py` before `model.init_state(...)`. Publication
pins both the commit and the diff fingerprint so the shared source matches the
production runtime rather than silently replacing it with clean upstream.

The production job uses `--reuse-detection-across-segments` and
`--sam3-detection-only`, so its effective detector batch is the live parser
default `--prepass-predict-batch=16`.

| Standalone config | Value | Production source |
|---|---:|---|
| `output_fps` | `10.0` | `--frames-fps 10` |
| `jpg_quality` | `90` | `--jpg-quality 90` |
| `detector_confidence` | `0.2` | `--yolo-detector-conf 0.2` |
| `detector_batch_size` | `16` | `--prepass-predict-batch` parser default |
| `sam3_input_max_side` | `1008` | `--sam3-input-max-side 1008` |
| `sam3_score_thresh` | `0.75` | `SAM3_PROMPT_YOLO_BBOX_THRESH` |
| `sam3_bottom_y2_frac` | `2.0` | `--sam3-point-bottom-y2-frac 2.0` |
| `sam3_window_sec` | `5.0` | `SAM3_POINT_WINDOW_SEC` |
| `sam3_point_expand` | `1.1` | `--sam3-point-expand 1.1` |
| `sam3_prompt_text` | `hand` | `--sam3-prompt-text hand` |
| `sam3_tracking_mode` | `joint` | `SAM3_POINT_TRACKING_MODE` |
| `sam3_init_mode` | `point` | `SAM3_POINT_INIT_MODE` |
| `sam3_prompt_geometry` | `box_below` | `SAM3_POINT_PROMPT_GEOMETRY` |
| `sam3_propagation_scheme` | `single_anchor` | `SAM3_POINT_PROPAGATION_SCHEME` |
| `sam3_mask_archive` | `npz` | `--sam3-mask-archive npz` |
| `sam3_ego_gate_enabled` | `true` | `SAM3_EGO_GATE_ENABLED=1` |
| `ego_gate_size_ratio_min` | `0.045` | matching production env |
| `ego_gate_size_ratio_good` | `0.187` | matching production env |
| `ego_gate_bottom_ratio_min` | `0.35` | matching production env |
| `ego_gate_bottom_ratio_good` | `0.85` | matching production env |
| `ego_gate_area_weight` | `0.7` | matching production env |
| `ego_gate_bottom_weight` | `0.3` | matching production env |
| `ego_gate_keep_tau` | `0.75` | matching production env |
| `ego_gate_flag_tau` | `0.35` | matching production env |
| `ego_gate_margin_tau` | `1.0` | matching production env |

The standalone defaults intentionally keep offline wandb provenance and visual
QA enabled. Production disables wandb and runs inside a larger scheduler; those
orchestration differences do not change detection or SAM3 masks.
