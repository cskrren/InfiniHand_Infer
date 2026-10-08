#!/usr/bin/env python3
"""Export only final per-side masks and detection metadata for POS upload."""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


SIDES = ("left", "right")
BOX_SCALE = 1.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-output", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--scene-uri", required=True)
    parser.add_argument("--run-tag", required=True)
    return parser.parse_args()


def tight_mask_box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.asarray(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32
    )


def expand_and_clip_box(
    box: np.ndarray, image_size_hw: tuple[int, int]
) -> np.ndarray:
    height, width = image_size_hw
    center = (box[:2] + box[2:]) * 0.5
    size = (box[2:] - box[:2]) * BOX_SCALE
    expanded = np.concatenate([center - size * 0.5, center + size * 0.5])
    expanded[[0, 2]] = np.clip(expanded[[0, 2]], 0.0, float(width - 1))
    expanded[[1, 3]] = np.clip(expanded[[1, 3]], 0.0, float(height - 1))
    return expanded.astype(np.float32)


def source_frame_id(stem: str):
    return np.int64(int(stem)) if stem.isdigit() else np.asarray(stem)


def destination_stem(source_name: str) -> str:
    """Match dataset convention: numeric det/mask names use at least 5 digits."""
    stem = Path(source_name).stem
    return f"{int(stem):05d}" if stem.isdigit() else stem


def main() -> int:
    args = parse_args()
    manifest = json.loads(
        (args.pipeline_output / "frame_manifest.json").read_text(encoding="utf-8")
    )
    pipeline_result = json.loads(
        (args.pipeline_output / "summary.json").read_text(encoding="utf-8")
    )
    run_config = json.loads(
        (args.pipeline_output / "run_config.json").read_text(encoding="utf-8")
    )
    num_frames = int(manifest["num_frames"])
    image_size_hw = (int(manifest["height"]), int(manifest["width"]))
    detection_root = args.export_dir / "detections"
    mask_root = args.export_dir / "masks"
    for side in SIDES:
        (detection_root / side).mkdir(parents=True, exist_ok=False)
        (mask_root / side).mkdir(parents=True, exist_ok=False)

    nonempty = {side: 0 for side in SIDES}
    for frame_index, frame_row in enumerate(manifest["frames"]):
        source_name = frame_row.get("source_file", frame_row["file"])
        source_stem = Path(source_name).stem
        stem = destination_stem(source_name)
        for side in SIDES:
            source_mask = (
                args.pipeline_output / "masks" / side / f"{frame_index:06d}.png"
            )
            mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(source_mask)
            mask = mask > 0
            destination_mask = mask_root / side / f"{stem}.png"
            shutil.copyfile(source_mask, destination_mask)

            valid = bool(mask.any())
            if valid:
                mask_box = tight_mask_box(mask)
                bbox = expand_and_clip_box(mask_box, image_size_hw)
                bbox_xywh = np.asarray(
                    [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    dtype=np.float32,
                )
                nonempty[side] += 1
                status = "ok"
            else:
                mask_box = np.full(4, np.nan, dtype=np.float32)
                bbox = np.full(4, np.nan, dtype=np.float32)
                bbox_xywh = np.full(4, np.nan, dtype=np.float32)
                status = "empty_mask"

            np.savez_compressed(
                detection_root / side / f"{stem}.npz",
                frame_index=np.int64(frame_index),
                source_frame_id=source_frame_id(source_stem),
                side_name=np.asarray(side),
                valid=np.bool_(valid),
                bbox_xyxy=bbox,
                bbox_xywh=bbox_xywh,
                mask_box_xyxy=mask_box,
                box_scale_hw=np.float32(BOX_SCALE),
                image_size_hw=np.asarray(image_size_hw, dtype=np.int32),
                sam_status=np.asarray(status),
                det_score=np.float32(np.nan),
                sam_prob=np.float32(np.nan),
                mask_area=np.int64(mask.sum()),
                source=np.asarray("step2_sam3_poss"),
            )

    summary = {
        "status": "ok",
        "scene_uri": args.scene_uri,
        "run_tag": args.run_tag,
        "num_frames": num_frames,
        "image_size_hw": list(image_size_hw),
        "detection_npz_count": num_frames * len(SIDES),
        "mask_png_count": num_frames * len(SIDES),
        "nonempty_masks": nonempty,
        "intermediate_storage": "/dev/shm",
        "sam3_tracking_mode": run_config["sam3_tracking_mode"],
        "pipeline_metrics": pipeline_result["metrics"],
        "pipeline_timings": pipeline_result["timings"],
    }
    for root in (detection_root, mask_root):
        (root / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
