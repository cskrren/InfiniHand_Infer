#!/usr/bin/env python3
"""Repair frames missing any per-side detection or mask in one POS dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


DATASETS = (
    "arctic", "dexycb", "egodex", "freihand", "hot3d", "indego",
    "interhand", "xperience10m", "egoverse", "h2o3d", "ho3d", "egolive",
)
SIDES = ("left", "right")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--rclone", default="/mnt/petrelfs/renkerui/.local/bin/rclone")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--detector-weight", type=Path)
    parser.add_argument("--sam3-checkpoint", type=Path)
    parser.add_argument("--sam3-src", type=Path)
    parser.add_argument("--remote-root", default="poss:ego/renkerui/ego_data")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def emit(event: str, **payload) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    emit("command", argv=command)
    subprocess.run(command, check=True, env=env)


def lsf(rclone: str, uri: str, *, dirs: bool = False) -> list[str]:
    command = [rclone, "lsf", uri, "--max-depth", "1"]
    command.append("--dirs-only" if dirs else "--files-only")
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def canonical_stem(filename: str) -> str:
    stem = Path(filename).stem
    return f"{int(stem):05d}" if stem.isdigit() else stem


def list_existing_stems(rclone: str, uri: str, suffix: str) -> set[str]:
    return {
        Path(name).stem
        for name in lsf(rclone, uri)
        if Path(name).suffix.lower() == suffix
    }


def scan_scene(
    rclone: str, scene_uri: str
) -> tuple[list[str], dict[str, list[str]], int]:
    image_names = sorted(
        name
        for name in lsf(rclone, f"{scene_uri}/images/")
        if Path(name).suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_names:
        raise RuntimeError(f"no images found: {scene_uri}/images/")
    output_stems = [canonical_stem(name) for name in image_names]
    if len(set(output_stems)) != len(output_stems):
        raise RuntimeError(f"canonical frame-name collision in {scene_uri}")

    existing = {}
    for side in SIDES:
        existing[("detections", side)] = list_existing_stems(
            rclone, f"{scene_uri}/detections/{side}/", ".npz"
        )
        existing[("masks", side)] = list_existing_stems(
            rclone, f"{scene_uri}/masks/{side}/", ".png"
        )

    missing_names = []
    reasons = {}
    for image_name, stem in zip(image_names, output_stems):
        frame_reasons = []
        for kind in ("detections", "masks"):
            for side in SIDES:
                if stem not in existing[(kind, side)]:
                    frame_reasons.append(f"{kind}/{side}")
        if frame_reasons:
            missing_names.append(image_name)
            reasons[image_name] = frame_reasons
    return missing_names, reasons, len(image_names)


def verify_scratch(export_dir: Path, image_names: list[str]) -> None:
    expected = {canonical_stem(name) for name in image_names}
    for side in SIDES:
        det = {path.stem for path in (export_dir / "detections" / side).glob("*.npz")}
        mask = {path.stem for path in (export_dir / "masks" / side).glob("*.png")}
        if det != expected:
            raise RuntimeError(f"scratch detection mismatch side={side}: {det ^ expected}")
        if mask != expected:
            raise RuntimeError(f"scratch mask mismatch side={side}: {mask ^ expected}")


def verify_remote(rclone: str, scene_uri: str, image_names: list[str]) -> None:
    expected = {canonical_stem(name) for name in image_names}
    for side in SIDES:
        det = list_existing_stems(rclone, f"{scene_uri}/detections/{side}/", ".npz")
        mask = list_existing_stems(rclone, f"{scene_uri}/masks/{side}/", ".png")
        if not expected.issubset(det):
            raise RuntimeError(f"remote detections still missing side={side}: {expected - det}")
        if not expected.issubset(mask):
            raise RuntimeError(f"remote masks still missing side={side}: {expected - mask}")


def repair_scene(
    args: argparse.Namespace, scene_uri: str, image_names: list[str], index: int
) -> None:
    scene_work = args.work_root / f"scene_{index:06d}"
    input_dir = scene_work / "images"
    input_dir.mkdir(parents=True, exist_ok=False)
    file_list = scene_work / "missing_images.txt"
    file_list.write_text("".join(f"{name}\n" for name in image_names), encoding="utf-8")
    try:
        run([
            args.rclone, "copy", f"{scene_uri}/images/", str(input_dir),
            "--files-from-raw", str(file_list), "--no-traverse",
            "--transfers", "16", "--checkers", "16", "--ignore-times",
        ])
        downloaded = {
            path.name for path in input_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        }
        if downloaded != set(image_names):
            raise RuntimeError(f"downloaded image mismatch: {downloaded ^ set(image_names)}")

        size_groups: dict[tuple[int, int], list[Path]] = defaultdict(list)
        for path in sorted(input_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            with Image.open(path) as image:
                size_groups[image.size].append(path)
        emit(
            "scene_resolution_groups",
            groups=[
                {"width": size[0], "height": size[1], "frames": len(paths)}
                for size, paths in sorted(size_groups.items())
            ],
        )

        env = os.environ.copy()
        pythonpath = [str(args.project_dir / "src"), str(args.sam3_src)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        for group_index, (_, paths) in enumerate(sorted(size_groups.items())):
            group_root = scene_work / f"resolution_{group_index:03d}"
            group_input = group_root / "images"
            pipeline_dir = group_root / "pipeline"
            export_dir = group_root / "export"
            group_input.mkdir(parents=True)
            for source in paths:
                os.link(source, group_input / source.name)
            group_names = [path.name for path in paths]
            run([
                args.python, "-m", "hand_detection_sam3_mask", "frames",
                "--config", str(args.config), "--output-dir", str(pipeline_dir),
                "--detector-weight", str(args.detector_weight),
                "--sam3-checkpoint", str(args.sam3_checkpoint),
                "--sam3-src", str(args.sam3_src), "--frames-dir", str(group_input),
                "--device", "cuda", "--overwrite",
            ], env=env)
            run([
                args.python, str(args.project_dir / "scripts" / "export_poss_results.py"),
                "--pipeline-output", str(pipeline_dir), "--export-dir", str(export_dir),
                "--scene-uri", scene_uri,
                "--run-tag", f"repair_{os.environ.get('SLURM_JOB_ID', 'local')}",
            ])
            verify_scratch(export_dir, group_names)

            for kind in ("detections", "masks"):
                for side in SIDES:
                    run([
                        args.rclone, "copy", str(export_dir / kind / side),
                        f"{scene_uri}/{kind}/{side}/",
                        "--transfers", "16", "--checkers", "16", "--ignore-times",
                    ])
            verify_remote(args.rclone, scene_uri, group_names)
        verify_remote(args.rclone, scene_uri, image_names)
    finally:
        shutil.rmtree(scene_work, ignore_errors=True)

def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    args.project_dir = args.project_dir.resolve()
    args.config = args.config or args.project_dir / "configs" / "poss_scene.yaml"
    args.detector_weight = args.detector_weight or args.project_dir / "weights" / "detector.pt"
    args.sam3_checkpoint = (
        args.sam3_checkpoint
        or args.project_dir / "checkpoints" / "facebook_sam3" / "sam3.pt"
    )
    args.sam3_src = args.sam3_src or args.project_dir / "third_party" / "sam3"
    args.work_root.mkdir(parents=True, exist_ok=False)
    dataset_uri = f"{args.remote_root.rstrip('/')}/{args.dataset}"
    records = []
    failures = []
    images_scanned = 0
    missing_frames = 0
    repaired_frames = 0
    try:
        scenes = [
            name.rstrip("/") for name in lsf(args.rclone, f"{dataset_uri}/", dirs=True)
        ]
        if args.max_scenes > 0:
            scenes = scenes[:args.max_scenes]
        emit("dataset_start", dataset=args.dataset, scenes=len(scenes), dry_run=args.dry_run)
        for index, scene in enumerate(scenes):
            scene_uri = f"{dataset_uri}/{scene}"
            try:
                missing_names, reasons, image_count = scan_scene(args.rclone, scene_uri)
                images_scanned += image_count
                missing_frames += len(missing_names)
                record = {
                    "scene": scene,
                    "image_count": image_count,
                    "missing_frames": len(missing_names),
                }
                records.append(record)
                if not missing_names:
                    emit("scene_complete", **record)
                    continue
                emit(
                    "scene_missing", **record,
                    examples={name: reasons[name] for name in missing_names[:5]},
                )
                if not args.dry_run:
                    repair_scene(args, scene_uri, missing_names, index)
                    repaired_frames += len(missing_names)
                    emit("scene_repaired", scene=scene, frames=len(missing_names))
            except Exception as error:
                failure = {"scene": scene, "error": f"{type(error).__name__}: {error}"}
                failures.append(failure)
                emit("scene_failed", **failure)

        summary = {
            "status": "failed" if failures else ("dry_run" if args.dry_run else "ok"),
            "dataset": args.dataset,
            "dataset_uri": dataset_uri,
            "scenes_scanned": len(scenes),
            "images_scanned": images_scanned,
            "missing_frames": missing_frames,
            "repaired_frames": repaired_frames,
            "failed_scenes": failures,
            "scene_records": records,
        }
        write_summary(args.summary_path, summary)
        emit(
            "dataset_complete",
            **{key: value for key, value in summary.items() if key != "scene_records"},
        )
        return 1 if failures else 0
    finally:
        shutil.rmtree(args.work_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
