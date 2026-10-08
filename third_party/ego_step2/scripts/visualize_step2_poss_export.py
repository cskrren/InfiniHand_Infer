#!/usr/bin/env python3
"""Visualize exported Step2 detections and masks against POS scene RGB frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SIDES = ("left", "right")
COLORS = {"left": (255, 80, 40), "right": (40, 80, 255)}  # BGR


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--contact-frames", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def load_detection(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as item:
        return {key: item[key] for key in item.files}


def draw_detection(image: np.ndarray, side: str, det: dict) -> None:
    if not bool(np.asarray(det["valid"]).item()):
        return
    box = np.asarray(det["bbox_xyxy"], dtype=np.float32)
    if not np.isfinite(box).all():
        return
    x1, y1, x2, y2 = np.rint(box).astype(int).tolist()
    color = COLORS[side]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    area = int(np.asarray(det.get("mask_area", 0)).item())
    cv2.putText(
        image,
        f"{side} area={area}",
        (x1, max(22, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )


def add_mask(image: np.ndarray, mask: np.ndarray, side: str, alpha: float) -> None:
    foreground = mask > 0
    if not foreground.any():
        return
    color = np.asarray(COLORS[side], dtype=np.float32)
    image[foreground] = (
        image[foreground].astype(np.float32) * (1.0 - alpha) + color * alpha
    ).astype(np.uint8)


def header(image: np.ndarray, title: str, frame_stem: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (250, 250, 250), 1, cv2.LINE_AA)
    cv2.putText(out, f"frame {frame_stem}", (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (215, 220, 230), 1, cv2.LINE_AA)
    return out


def resize_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    return cv2.resize(image, (width, max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def main() -> int:
    cfg = args()
    images_dir = cfg.input_root / "images"
    detections_dir = cfg.input_root / "detections"
    masks_dir = cfg.input_root / "masks"
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    (cfg.output_dir / "frames").mkdir()

    image_paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"no RGB frames under {images_dir}")
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(image_paths[0])
    height, width = first.shape[:2]
    video_path = cfg.output_dir / "rgb_detection_mask_triptych.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps, (width * 3, height)
    )
    if not writer.isOpened():
        raise RuntimeError(video_path)

    contact_indices = set(
        np.linspace(0, len(image_paths) - 1, min(cfg.contact_frames, len(image_paths)), dtype=np.int64).tolist()
    )
    contact_tiles = []
    stats = {side: {"valid_detections": 0, "nonempty_masks": 0, "mask_pixels": 0} for side in SIDES}
    missing = []
    for index, image_path in enumerate(image_paths):
        stem = image_path.stem
        rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(image_path)
        if rgb.shape[:2] != (height, width):
            raise ValueError(f"inconsistent RGB size: {image_path} {rgb.shape[:2]}")
        detection_panel = rgb.copy()
        overlay_panel = rgb.copy()
        for side in SIDES:
            det_path = detections_dir / side / f"{stem}.npz"
            mask_path = masks_dir / side / f"{stem}.png"
            if not det_path.is_file() or not mask_path.is_file():
                missing.append({"frame": stem, "side": side})
                continue
            det = load_detection(det_path)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (height, width):
                raise ValueError(f"bad mask {mask_path}: {None if mask is None else mask.shape}")
            if bool(np.asarray(det["valid"]).item()):
                stats[side]["valid_detections"] += 1
            pixels = int((mask > 0).sum())
            if pixels:
                stats[side]["nonempty_masks"] += 1
                stats[side]["mask_pixels"] += pixels
            draw_detection(detection_panel, side, det)
            add_mask(overlay_panel, mask, side, cfg.alpha)
            draw_detection(overlay_panel, side, det)
        triptych = np.concatenate(
            [header(rgb, "RGB", stem), header(detection_panel, "RGB + detections", stem), header(overlay_panel, "RGB + detections + masks", stem)],
            axis=1,
        )
        writer.write(triptych)
        if index in contact_indices:
            tile = resize_width(triptych, 1440)
            contact_tiles.append(tile)
            cv2.imwrite(str(cfg.output_dir / "frames" / f"{stem}_triptych.jpg"), triptych)
    writer.release()

    columns = 2
    tile_h, tile_w = contact_tiles[0].shape[:2]
    rows = (len(contact_tiles) + columns - 1) // columns
    sheet = np.zeros((rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
    for index, tile in enumerate(contact_tiles):
        row, column = divmod(index, columns)
        sheet[row * tile_h:(row + 1) * tile_h, column * tile_w:(column + 1) * tile_w] = tile
    contact_path = cfg.output_dir / "contact_sheet.jpg"
    if not cv2.imwrite(str(contact_path), sheet):
        raise RuntimeError(contact_path)

    summary = {
        "status": "ok" if not missing else "missing_pairs",
        "num_rgb_frames": len(image_paths),
        "frame_size_hw": [height, width],
        "stats": stats,
        "missing_pairs": missing,
        "outputs": {"video": str(video_path), "contact_sheet": str(contact_path)},
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
