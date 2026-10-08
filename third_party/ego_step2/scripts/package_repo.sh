#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
[ -z "$(git status --porcelain)" ]
COMMIT=$(git rev-parse --short=12 HEAD)
OUTPUT=${1:-"$PROJECT_DIR/../hand_detection_sam3_mask_${COMMIT}.tar.gz"}
git archive --format=tar.gz --prefix=hand_detection_sam3_mask/ HEAD -o "$OUTPUT"
shasum -a 256 "$OUTPUT"
